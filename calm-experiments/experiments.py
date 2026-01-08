# experiments/experiment_runner.py
import importlib
import yaml
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
import os
import time
import warnings
import torch
from utils import metrics, logging_utils
from effector.calm.datasets.base import Dataset
from effector.calm.calm import (
    CALM_CLASSIFICATION_MODELS,
    CALM_REGRESSION_MODELS,
    RegionalPDPDetector,
    RegionalRHALEDetector,
)
from effector.calm.masked_fitting import (
    MASKED_GAM_CLASSIFICATION_MODELS,
    MASKED_GAM_REGRESSION_MODELS,
)
from effector.calm.competitors import (
    COMPETITOR_CLASSIFICATION_MODELS,
    COMPETITOR_REGRESSION_MODELS,
)
from effector.calm.blackbox import (
    CLASSIFICATION_DATASETS_BLACKBOX,
    REGRESSION_DATASETS_BLACKBOX,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_random_seeds(seed=None):
    import numpy as np
    import tensorflow as tf
    import random

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentRunner:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.stop_on_error = self.config.get("stop_on_error", False)
        self.random_seed = self.config.get("random_seed", 42)
        self.test_size = self.config.get("test_size", 0.2)
        self.output_file = self.config.get("output_file", "experiments/results.csv")
        self.logger = logging_utils.get_logger(__name__)
        self.kfold_n_splits = self.config.get("kfold_n_splits", 1)
        self.expected_columns = [
            "dataset",
            "task",
            "method",
            "random_seed",
            "test_size",
            "region_detector",
            "pcg_drop_thres",
            "nof_splits_num",
            "masked_gam",
            "blackbox_model",
        ]
        # Add metric keys from config.
        metric_keys = self.config.get("metrics", []) + [
            "runtime_sec",
            "num_interactions",
        ]
        if self.kfold_n_splits > 1:
            metric_keys = [m + "_mean" for m in metric_keys] + [
                m + "_std" for m in metric_keys
            ]

        self.expected_columns.extend(metric_keys)

        # Initialize results list.
        # If append_to_file is True and file exists, load it and check columns.
        if self.config.get("append_to_file", False) and os.path.exists(
            self.output_file
        ):
            self.logger.info(f"Appending to existing file {self.output_file}")
            existing_df = pd.read_csv(self.output_file)
            if list(existing_df.columns) != self.expected_columns:
                raise ValueError(
                    f"Existing file columns {list(existing_df.columns)} do not match expected columns {self.expected_columns}."
                )
            self.results = existing_df.to_dict(orient="records")
        else:
            self.results = []

    def _load_class(self, module_path, class_name):
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _instantiate_dataset(self, dataset_config) -> Dataset:
        print(f"Loading dataset module: {dataset_config['module']}")
        module = importlib.import_module(dataset_config["module"])
        dataset_class = getattr(module, dataset_config["name"])
        return dataset_class()

    def _instantiate_model(self, method_config, param_combination):
        model_class_sets = {
            "maskedgam": {
                "classification": MASKED_GAM_CLASSIFICATION_MODELS,
                "regression": MASKED_GAM_REGRESSION_MODELS,
            },
            "calm": {
                "classification": CALM_CLASSIFICATION_MODELS,
                "regression": CALM_REGRESSION_MODELS,
            },
            "competitor": {
                "classification": COMPETITOR_CLASSIFICATION_MODELS,
                "regression": COMPETITOR_REGRESSION_MODELS,
            },
            "blackbox": {
                "classification": CLASSIFICATION_DATASETS_BLACKBOX,
                "regression": REGRESSION_DATASETS_BLACKBOX,
            },
        }

        blackbox_args = {}
        if method_config["type"] == "blackbox":
            return model_class_sets["blackbox"][self.config["task"]][
                method_config["name"]
            ]()

        model_class_set = model_class_sets[method_config["type"]][self.config["task"]]
        ModelClass = model_class_set[method_config["name"]]

        if method_config["type"] == "calm":
            if param_combination["region_detector"] == "RegionalPDP":
                region_detector = RegionalPDPDetector(
                    heter_pcg_threshold=param_combination["pcg_drop_thres"],
                    nof_splits_numerical=param_combination["nof_splits_num"],
                )
            elif param_combination["region_detector"] == "RegionalRHALE":
                region_detector = RegionalRHALEDetector(
                    heter_pcg_threshold=param_combination["pcg_drop_thres"],
                    nof_splits_numerical=param_combination["nof_splits_num"],
                )
            else:
                raise ValueError(
                    f"Unsupported region detector: {param_combination['region_detector']}"
                )

            return ModelClass(
                masked_gam_name=param_combination["masked_gam_name"],
                masked_gam_args=param_combination.get("masked_gam_args", {}),
                blackbox_model=model_class_sets["blackbox"][self.config["task"]][
                    param_combination["blackbox_model"]
                ],
                blackbox_args=blackbox_args,
                region_detector=region_detector,
            )

        return ModelClass(**param_combination)

    def _update_results_file(self):
        # Create DataFrame from self.results and reindex columns to expected_columns.
        df = pd.DataFrame(self.results)
        df = df.reindex(columns=self.expected_columns)
        df.to_csv(self.output_file, index=False)

    def get_meta_info(self, X, y, features, numerical_features, task):
        meta_info = {}
        X = X.copy()
        y = y.copy()
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        for i, feat in enumerate(features):
            if feat in numerical_features:
                sx = MinMaxScaler((0, 1))
                sx.fit([[0], [1]])
                X[:, [i]] = sx.transform(X[:, [i]])
                meta_info[feat] = {"scaler": sx, "type": "continuous"}
            else:
                meta_info[feat] = {"type": "categorical"}
                sx = OrdinalEncoder()
                X[:, [i]] = sx.fit_transform(X[:, [i]])
                meta_info[feat]["values"] = []
                for item in sx.categories_[0].tolist():
                    try:
                        if item == int(item):
                            meta_info[feat]["values"].append(str(int(item)))
                        else:
                            meta_info[feat]["values"].append(str(item))
                    except ValueError:
                        meta_info[feat]["values"].append(str(item))

        if isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1)

        if task == "regression":
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info["Y"] = {"scaler": sy, "type": "target"}
        else:
            sy = None
            meta_info["Y"] = {"type": "target"}
        X = X.astype(np.float32)
        return meta_info, X, y, sy

    def run(self):
        for dataset_config in self.config["datasets"]:
            set_random_seeds(self.random_seed)
            dataset = self._instantiate_dataset(dataset_config)
            task = self.config["task"]
            self.logger.info(
                f"Processing dataset: {dataset_config['name']} (task: {task})"
            )
            try:
                dataset.fetch()
                dataset.preprocess()
                X, y = dataset.get_Xy()
            except Exception as e:
                self.logger.error(
                    f"Error processing dataset {dataset_config['name']}: {str(e)}"
                )
                if self.stop_on_error:
                    raise
                else:
                    continue

            features, numerical_features, categorical_features = (
                dataset.get_feature_names()
            )
            meta_info, minmax_sc_X, minmax_sc_y, minmax_sc = self.get_meta_info(
                X, y, features, numerical_features, task
            )
            t = Pipeline(
                [
                    (
                        "ord",
                        ColumnTransformer(
                            [("cat", OrdinalEncoder(), categorical_features)],
                            remainder="passthrough",
                        ),
                    ),
                    ("std", StandardScaler()),
                ]
            )
            X = t.fit_transform(X)

            sd_sc = None
            if task == "regression":
                sd_sc = StandardScaler()
                y = sd_sc.fit_transform(y.values.reshape(-1, 1)).ravel()

            if self.kfold_n_splits > 1:
                global_cv = (
                    StratifiedKFold(
                        n_splits=self.kfold_n_splits,
                        random_state=self.random_seed,
                        shuffle=True,
                    )
                    if task == "classification"
                    else KFold(
                        n_splits=self.kfold_n_splits,
                        random_state=self.random_seed,
                        shuffle=True,
                    )
                )
                X_train, X_test, y_train, y_test = None, None, None, None
                X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = (
                    None,
                    None,
                    None,
                    None,
                )
            else:
                global_cv = None
                stratify = y if task == "classification" else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=self.random_seed,
                    stratify=stratify,
                )
                X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = (
                    train_test_split(
                        minmax_sc_X,
                        minmax_sc_y,
                        test_size=self.test_size,
                        random_state=self.random_seed,
                        stratify=stratify,
                    )
                )

            for method_config in self.config["methods"]:
                method_name = method_config["name"]
                param_grid = method_config.get("parameters", {})
                param_combinations = list(ParameterGrid(param_grid=param_grid))

                self.logger.info(f"param_grid: {param_grid}")
                self.logger.info(f"param_combinations: {param_combinations}")
                for params in param_combinations:

                    if method_name in [
                        "CALMClassifier",
                        "CALMRegressor",
                    ]:
                        params_keys = list(params.keys())

                        if "blackbox_model" not in params_keys:
                            params["blackbox_model"] = (
                                "XGBClassifier"
                                if task == "classification"
                                else "XGBRegressor"
                            )

                        if "masked_gam_name" not in params_keys:
                            params["masked_gam_name"] = (
                                "NoInteractionsEBMRegressor"
                                if task == "regression"
                                else "NoInteractionsEBMClassifier"
                            )

                        if "region_detector" not in params_keys:
                            params["region_detector"] = "RegionalPDP"

                        if "pcg_drop_thres" not in params_keys:
                            params["pcg_drop_thres"] = 0.2

                        if "nof_splits_num" not in params_keys:
                            params["nof_splits_num"] = 20

                    if (
                        method_config["type"] == "calm"
                        and params["region_detector"] == "RegionalRHALE"
                        and params["blackbox_model"]
                        in [
                            "RFClassifier",
                            "RFRegressor",
                            "XGBClassifier",
                            "XGBRegressor",
                        ]
                    ):
                        continue

                    self.logger.info(
                        f"Running method {method_name} with parameters {params} on dataset {dataset_config['name']}"
                    )
                    try:
                        # For standalone masked GAMs, set the 'dim' parameter for instantiation and then remove it.
                        if method_name in [
                            "MaskedNAMRegressor",
                            "MaskedNAMClassifier",
                            "PyGAMRegressor",
                            "PyGAMClassifier",
                        ]:
                            params["dim"] = X.shape[1]
                        is_gaminet = method_name in [
                            "GAMINetRegressor",
                            "GAMINetClassifier",
                        ]
                        if is_gaminet:
                            params["task_type"] = (
                                "Classification"
                                if task == "classification"
                                else "Regression"
                            )
                            params["meta_info"] = meta_info
                        self._evaluate_model(
                            method_name,
                            method_config,
                            params,
                            dataset_config,
                            task,
                            minmax_sc_X if is_gaminet else X,
                            minmax_sc_y if is_gaminet else y,
                            global_cv,
                            X_train_minmax if is_gaminet else X_train,
                            X_test_minmax if is_gaminet else X_test,
                            y_train_minmax if is_gaminet else y_train,
                            y_test_minmax if is_gaminet else y_test,
                            minmax_sc if is_gaminet else sd_sc,
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error running {method_name} with params {params} on {dataset_config['name']}: {str(e)}"
                        )
                        self.logger.debug(traceback.format_exc())

                        if params.get("masked_gam_name", "") in [
                            "PyGAMRegressor",
                            "PyGAMClassifier",
                        ] and not params.get("use_grid_lam_search", False):
                            self.logger.info(
                                f"Retrying {method_name} with 'use_grid_lam_search=True' due to error."
                            )
                            try:
                                params["masked_gam_args"] = {
                                    "use_grid_lam_search": True
                                }

                                self._evaluate_model(
                                    method_name,
                                    method_config,
                                    params,
                                    dataset_config,
                                    task,
                                    minmax_sc_X if is_gaminet else X,
                                    minmax_sc_y if is_gaminet else y,
                                    global_cv,
                                    X_train_minmax if is_gaminet else X_train,
                                    X_test_minmax if is_gaminet else X_test,
                                    y_train_minmax if is_gaminet else y_train,
                                    y_test_minmax if is_gaminet else y_test,
                                    minmax_sc if is_gaminet else sd_sc,
                                )
                            except Exception as retry_e:
                                self.logger.error(
                                    f"Retry also failed for {method_name} on {dataset_config['name']} with error: {str(retry_e)}"
                                )
                                self.logger.debug(traceback.format_exc())
                                if self.stop_on_error:
                                    raise
                        elif self.stop_on_error:
                            raise
                        else:
                            continue

        self.logger.info(
            f"All experiments completed. Results are in {self.output_file}"
        )

    def _evaluate_model(
        self,
        method_name,
        method_config,
        params,
        dataset_config,
        task,
        X,
        y,
        global_cv,
        X_train,
        X_test,
        y_train,
        y_test,
        sc,
    ):
        if self.kfold_n_splits > 1:
            kfolds_results = {m: [] for m in self.config["metrics"]}
            kfolds_results["runtime_sec"] = []
            kfolds_results["num_interactions"] = []

            for fold_index, (train_index, test_index) in enumerate(
                global_cv.split(X, y)
            ):
                set_random_seeds(self.random_seed)
                model = self._instantiate_model(method_config, params)

                self.logger.info(f"Model instantiated: {model}")
                X_train_fold, X_test_fold = X[train_index], X[test_index]
                y_train_fold, y_test_fold = y[train_index], y[test_index]

                fold_metrics_results = self._run_train_test(
                    model,
                    X_train_fold,
                    X_test_fold,
                    y_train_fold,
                    y_test_fold,
                    task,
                    method_config["type"],
                    sc,
                )
                for m, v in fold_metrics_results.items():
                    kfolds_results[m].append(v)

            kfolds_results_means = {
                m + "_mean": np.mean(v) for m, v in kfolds_results.items()
            }
            kfolds_results_stds = {
                m + "_std": np.std(v) for m, v in kfolds_results.items()
            }
            metrics_results = {**kfolds_results_means, **kfolds_results_stds}
        else:
            set_random_seeds(self.random_seed)
            model = self._instantiate_model(method_config, params)

            self.logger.info(f"Model instantiated: {model}")
            metrics_results = self._run_train_test(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                task,
                method_config["type"],
                sc,
            )

        self.logger.info(f"Run finished with metrics: {metrics_results}")

        result = {
            "dataset": dataset_config["name"],
            "task": task,
            "method": method_name,
            "random_seed": self.random_seed,
            "test_size": self.test_size,
        }

        if method_config["type"] == "calm":
            result["region_detector"] = params.get("region_detector")
            result["pcg_drop_thres"] = params.get("pcg_drop_thres")
            result["nof_splits_num"] = params.get("nof_splits_num")
            result["masked_gam"] = params.get("masked_gam_name")
            result["blackbox_model"] = params.get("blackbox_model")

        params.pop("dim", None)
        result.update(params)
        result.update(metrics_results)

        self.results.append(result)
        self._update_results_file()

    def _run_train_test(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        task,
        method_type,
        sc=None,
    ):

        start_time = time.time()
        set_random_seeds(self.random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed_time = time.time() - start_time

        metric_results = {}
        if task == "classification":
            for m in self.config["metrics"]:
                score = metrics.compute_classification_metric(m, y_test, y_pred)
                metric_results[m] = score
        else:
            for m in self.config["metrics"]:
                score = metrics.compute_regression_metric(m, y_test, y_pred, sc)
                metric_results[m] = score
        metric_results["runtime_sec"] = elapsed_time
        if method_type == "calm":
            # For CALM, we count half the total number of leaves across all trees,
            # since each interaction (up to 4 regions at max depth) is visualizable in 1D,
            # in contrast to the interactions found by the EB2M methods.
            metric_results["num_interactions"] = model.interactions_info["leaf"] // 2

        if method_type == "competitor":
            from effector.calm.competitors import (
                EBM2Classifier,
                EBM2Regressor,
                NodeGAM2Classifier,
                NodeGAM2Regressor,
                BaseGAMINetModel,
            )

            if isinstance(model, (EBM2Classifier, EBM2Regressor)):
                metric_results["num_interactions"] = sum(
                    1 for term in model.model.term_features_ if len(term) > 1
                )
            elif isinstance(model, (NodeGAM2Classifier, NodeGAM2Regressor)):
                nodegam_model = model.model
                df = nodegam_model.model.extract_additive_terms(
                    X=pd.DataFrame(X_train),
                    norm_fn=nodegam_model.preprocessor.transform,
                    y_mu=nodegam_model.preprocessor.y_mu,
                    y_std=nodegam_model.preprocessor.y_std,
                    device=nodegam_model.device,
                    batch_size=2 * nodegam_model.batch_size,
                    purify=False,
                )
                num_interactions = (
                    df["feat_idx"].apply(lambda x: isinstance(x, tuple)).sum()
                )
                metric_results["num_interactions"] = num_interactions
            elif isinstance(model, BaseGAMINetModel):
                metric_results["num_interactions"] = len(model.model.interaction_list)
            else:
                warnings.warn(
                    f"Cannot compute number of regions for competitor model type: {model}"
                )
                metric_results["num_interactions"] = None
        return metric_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config_regression.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--stop_on_error", action="store_true", help="Stop if any error is encountered"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    if args.stop_on_error:
        runner.stop_on_error = True
    runner.run()


if __name__ == "__main__":
    main()
