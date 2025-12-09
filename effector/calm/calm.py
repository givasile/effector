from typing import Optional
from abc import abstractmethod, ABC
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import effector
from .masked_fitting import (
    NoInteractionsEBMClassifier,
    NoInteractionsEBMRegressor,
    WithInteractionsEBMClassifier,
    WithInteractionsEBMRegressor,
    MaskedNAMClassifier,
    MaskedNAMRegressor,
    PyGAMClassifier,
    PyGAMRegressor,
)
from .blackbox import (
    BlackBoxModel,
    XGBClassifier,
    XGBRegressor,
)


class RegionDetector(ABC):
    @abstractmethod
    def detect_regions(self, X, blackbox_model) -> dict:
        """
        Given data and predictions (or feature effects), returns region definitions.
        """
        pass


class RegionalPDPDetector(RegionDetector):
    def __init__(
        self,
        heter_pcg_threshold,
        nof_splits_numerical,
        max_depth=2,
    ):
        self.heter_pcg_threshold = heter_pcg_threshold
        self.nof_splits_numerical = nof_splits_numerical
        self.max_depth = max_depth

    def detect_regions(
        self,
        X,
        blackbox_model: BlackBoxModel,
    ):
        self.regional_fe = effector.RegionalPDP(
            data=X,
            model=blackbox_model.forward,
            cat_limit=10,
            nof_instances="all",
        )
        self.regional_fe.fit(
            features="all",
            candidate_conditioning_features="all",
            space_partitioner=effector.space_partitioning.Best(
                min_heterogeneity_decrease_pcg=self.heter_pcg_threshold,
                numerical_features_grid_size=self.nof_splits_numerical,
                max_depth=self.max_depth,
            ),
        )

        tree = self.regional_fe.tree
        return tree


class RegionalRHALEDetector(RegionDetector):
    def __init__(
        self,
        heter_pcg_threshold,
        nof_splits_numerical,
    ):
        self.heter_pcg_threshold = heter_pcg_threshold
        self.nof_splits_numerical = nof_splits_numerical

    def detect_regions(
        self,
        X,
        blackbox_model: BlackBoxModel,
    ):
        self.regional_fe = effector.RegionalRHALE(
            data=X,
            model=blackbox_model.forward,
            model_jac=blackbox_model.jac,
            cat_limit=10,
            nof_instances="all",
        )
        self.regional_fe.fit(
            features="all",
            candidate_conditioning_features="all",
            space_partitioner=effector.space_partitioning.Best(
                min_heterogeneity_decrease_pcg=self.heter_pcg_threshold,
                numerical_features_grid_size=self.nof_splits_numerical,
            ),
        )

        tree = self.regional_fe.tree
        return tree


class CALMBase(BaseEstimator, ABC):
    def __init__(
        self,
        masked_gam_name: str,
        masked_gam_args: Optional[dict] = {},
        blackbox_model: Optional["BlackBoxModel"] = None,
        blackbox_args: Optional[dict] = {},
        region_detector: Optional["RegionDetector"] = None,
        refit_blackbox: bool = True,
    ):
        """
        Parameters:
            blackbox_model: instance of BlackBoxModel
            region_detector: instance of RegionDetector
            masked_gam: instance of GAMModel
        """
        self.blackbox_model = blackbox_model
        self.blackbox_args = blackbox_args
        self.region_detector = region_detector
        self.masked_gam_name = masked_gam_name
        self.masked_gam_args = masked_gam_args
        self.masked_gam = None
        self.tree = None  # Will be set after region detection
        self.interactions_info = None
        self.refit_blackbox = refit_blackbox
        self.seed = 42

    def set_random_seeds(self):
        import numpy as np
        import tensorflow as tf
        import random
        import torch

        seed = self.seed

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def data_transform(self, X, tree, labels=None):

        def feature_transform(idx, data, tree_dict):
            if tree_dict is None:
                return (
                    [data[idx]],
                    [np.ones(data[idx, :].shape[0])],
                    ["x{}".format(idx)] if labels is None else labels[idx],
                )

            def create_mask(data, comp, pos, foc_index):
                if comp == "==":
                    return data[:, foc_index] == pos
                elif comp == "!=":
                    return data[:, foc_index] != pos
                elif comp == "<=":
                    return data[:, foc_index] <= pos
                elif comp == ">":
                    return data[:, foc_index] > pos
                elif comp == "<":
                    return data[:, foc_index] < pos
                elif comp == ">=":
                    return data[:, foc_index] >= pos
                else:
                    raise ValueError("Invalid comparison")

            mask_list = []
            new_col_names = []

            def recursive_feature_transform(node, mask, new_name):
                pos = node.info.get("foc_split_position", None)
                foc_index = node.info.get("foc_index", None)
                comp = node.info.get("comparison", None)

                if pos is not None:
                    cur_mask = create_mask(data, comp, pos, foc_index)
                    mask = mask.copy()
                    mask = np.logical_and(mask, cur_mask)
                    new_name += (
                        "x{} {} {:.2f} & ".format(foc_index, comp, pos)
                        if labels is None
                        else "{} {} {:.2f} & ".format(labels[foc_index], comp, pos)
                    )

                children = tree_dict.get_children(node.name)
                if len(children) == 0:
                    mask_list.append(mask)
                    new_col_names.append(new_name[:-3])
                    return

                for child in children:
                    recursive_feature_transform(child, mask, new_name)

            root = tree_dict.get_root()
            mask = np.ones(data.shape[0]) * True
            new_name = (
                "x{} | ".format(idx) if labels is None else "{} | ".format(labels[idx])
            )
            recursive_feature_transform(root, mask, new_name)

            mask = np.stack(mask_list, axis=1)
            new_data = np.repeat(
                data[:, idx, np.newaxis], repeats=mask.shape[1], axis=-1
            )
            new_data = new_data * mask

            return new_data, mask, new_col_names

        new_data = []
        all_mask = []
        new_names = []
        foi_2_foc = {}
        for i in range(X.shape[1]):
            data, mask, names = feature_transform(i, X, tree["feature_{}".format(i)])
            foi_2_foc[i] = list(range(len(new_names), len(new_names) + len(names)))
            new_data.append(data)
            all_mask.append(mask)
            new_names.extend(names)

        new_data = np.concatenate(new_data, axis=-1)
        all_mask = np.concatenate(all_mask, axis=-1).astype(float)
        self.foi_2_foc = foi_2_foc
        return new_data, all_mask, new_names

    def get_nodes_cnts(self, N):
        counts = {
            "level_1": 0,
            "level_2": 0,
            "leaf": 0,
            "foi": 0,
            "foc": 0,
        }

        def traverse(node, tree_dict):
            level = node.info.get("level")
            foc_index = node.info.get("foc_index", None)
            if level == 1:
                counts["level_1"] += 1
                foc_set.add(foc_index)
            elif level == 2:
                counts["level_2"] += 1
                foc_set.add(foc_index)

            children = tree_dict.get_children(node.name)
            if not children and level > 0:
                counts["leaf"] += 1

            for child in children:
                traverse(child, tree_dict)

        for i in range(N):
            tree_dict = self.tree.get(f"feature_{i}")
            root = tree_dict.get_root()
            foc_set = set()
            prev_leaf = counts["leaf"]

            traverse(root, tree_dict)

            counts["foc"] += len(foc_set)
            new_leaf = counts["leaf"]
            if new_leaf > prev_leaf:
                counts["foi"] += 1

        self.interactions_info = counts
        return counts

    @abstractmethod
    def fit(self, X, y) -> "CALMBase":
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def summary(self, features):
        pass

    def plot(self, feature, node_id):
        pass


class CALMClassifier(CALMBase, ClassifierMixin):
    def __init__(
        self,
        blackbox_model: Optional["BlackBoxModel"] = None,
        blackbox_args: Optional[dict] = {},
        region_detector: Optional["RegionDetector"] = None,
        masked_gam_name: Optional[str] = "NoInteractionsEBMClassifier",
        masked_gam_args: Optional[dict] = {},
        refit_blackbox: bool = True,
        feat_types: Optional[any] = None,
    ):
        self.masked_gam_constr_dict = {
            "MaskedNAMClassifier": MaskedNAMClassifier,
            "PyGAMClassifier": PyGAMClassifier,
            "NoInteractionsEBMClassifier": NoInteractionsEBMClassifier,
            "WithInteractionsEBMClassifier": WithInteractionsEBMClassifier,
        }
        self.feat_types = feat_types
        assert masked_gam_name in self.masked_gam_constr_dict.keys()
        super().__init__(
            blackbox_model=blackbox_model,
            blackbox_args=blackbox_args,
            region_detector=region_detector,
            masked_gam_name=masked_gam_name,
            masked_gam_args=masked_gam_args,
            refit_blackbox=refit_blackbox,
        )

    def fit(self, X, y, max_depth=2, feat_labels=None):
        if self.blackbox_model is None and not self.refit_blackbox:
            raise ValueError(
                "Refitting of blackbox model disabled, but no model was provided."
            )
        self.blackbox_model = (
            self.blackbox_model if self.blackbox_model is not None else XGBClassifier
        )
        self.blackbox_model_instance = self.blackbox_model(**self.blackbox_args)
        self.region_detector = (
            self.region_detector
            if self.region_detector is not None
            else RegionalPDPDetector(
                heter_pcg_threshold=0.2, nof_splits_numerical=20, max_depth=max_depth
            )
        )

        # 1. Fit the black-box model
        if self.refit_blackbox:
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Refitting blackbox model")
            self.blackbox_model_instance.fit(X, y)

        # 2. Detect regions using the region detection algorithm
        self.tree = self.region_detector.detect_regions(X, self.blackbox_model_instance)
        self.get_nodes_cnts(X.shape[1])

        # 3. Transform the dataset based on the regions (e.g., add new features)
        X_transformed, mask, new_names = self.data_transform(
            X, self.tree, labels=feat_labels
        )
        self.new_names = new_names

        self.set_random_seeds()
        # if self.masked_gam is None:
        if self.masked_gam_name in ["MaskedNAMClassifier", "PyGAMClassifier"]:
            self.masked_gam_args["dim"] = X_transformed.shape[1]

        if self.feat_types is not None and self.masked_gam_name in [
            "NoInteractionsEBMClassifier",
            "NoInteractionsEBMRegressor",
        ]:
            self.new_feat_types = np.full(
                X_transformed.shape[1], "unknown", dtype=object
            )
            for foi, foc in self.foi_2_foc.items():
                self.new_feat_types[foc] = self.feat_types[foi]
            self.masked_gam_args["feature_types"] = self.new_feat_types

        self.masked_gam = self.masked_gam_constr_dict[self.masked_gam_name](
            **self.masked_gam_args
        )

        # 4. Fit the GAM on the transformed dataset
        self.masked_gam.fit(X_transformed, y, mask=mask)
        return self

    def predict(self, X):
        X_transformed, mask, _ = self.data_transform(X, self.tree)
        y = self.masked_gam.predict(X_transformed, mask)

        # Convert to binary

        return (y > 0.5).astype(int)


class CALMRegressor(CALMBase, RegressorMixin):
    def __init__(
        self,
        blackbox_model: Optional["BlackBoxModel"] = None,
        blackbox_args: Optional[dict] = {},
        region_detector: Optional["RegionDetector"] = None,
        masked_gam_name: Optional[str] = "NoInteractionsEBMRegressor",
        masked_gam_args: Optional[dict] = {},
        refit_blackbox: bool = True,
        feat_types: Optional[any] = None,
    ):
        self.masked_gam_constr_dict = {
            "MaskedNAMRegressor": MaskedNAMRegressor,
            "PyGAMRegressor": PyGAMRegressor,
            "NoInteractionsEBMRegressor": NoInteractionsEBMRegressor,
            "WithInteractionsEBMRegressor": WithInteractionsEBMRegressor,
        }
        self.feat_types = feat_types
        assert masked_gam_name in self.masked_gam_constr_dict.keys()
        super().__init__(
            blackbox_model=blackbox_model,
            blackbox_args=blackbox_args,
            region_detector=region_detector,
            masked_gam_name=masked_gam_name,
            masked_gam_args=masked_gam_args,
            refit_blackbox=refit_blackbox,
        )

    def fit(self, X, y, max_depth=2, feat_labels=None):
        if self.blackbox_model is None and not self.refit_blackbox:
            raise ValueError(
                "Refitting of blackbox model disabled, but no model was provided."
            )
        self.blackbox_model = (
            self.blackbox_model if self.blackbox_model is not None else XGBRegressor
        )
        self.blackbox_model_instance = self.blackbox_model(**self.blackbox_args)
        self.region_detector = (
            self.region_detector
            if self.region_detector is not None
            else RegionalPDPDetector(
                heter_pcg_threshold=0.2, nof_splits_numerical=20, max_depth=max_depth
            )
        )

        # 1. Fit the black-box model
        if self.refit_blackbox:
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Refitting blackbox model")
            self.blackbox_model_instance.fit(X, y)

        # 2. Detect regions using the region detection algorithm
        self.tree = self.region_detector.detect_regions(X, self.blackbox_model_instance)
        self.get_nodes_cnts(X.shape[1])

        # 3. Transform the dataset based on the regions (e.g., add new features)
        X_transformed, mask, new_names = self.data_transform(
            X, self.tree, labels=feat_labels
        )
        self.new_names = new_names

        self.set_random_seeds()
        # if self.masked_gam is None:
        if self.masked_gam_name in ["MaskedNAMRegressor", "PyGAMRegressor"]:
            self.masked_gam_args["dim"] = X_transformed.shape[1]

        if self.feat_types is not None and self.masked_gam_name in [
            "NoInteractionsEBMClassifier",
            "NoInteractionsEBMRegressor",
        ]:
            self.new_feat_types = np.full(
                X_transformed.shape[1], "unknown", dtype=object
            )
            for foi, foc in self.foi_2_foc.items():
                self.new_feat_types[foc] = self.feat_types[foi]
            self.masked_gam_args["feature_types"] = self.new_feat_types

        self.masked_gam = self.masked_gam_constr_dict[self.masked_gam_name](
            **self.masked_gam_args
        )

        # 4. Fit the GAM on the transformed dataset
        self.masked_gam.fit(X_transformed, y, mask=mask)
        return self

    def predict(self, X):
        X_transformed, mask, _ = self.data_transform(X, self.tree)
        return self.masked_gam.predict(X_transformed, mask)


# Dictionaries registering all CALM models by task
CALM_CLASSIFICATION_MODELS = {
    "CALMClassifier": CALMClassifier,
}

CALM_REGRESSION_MODELS = {
    "CALMRegressor": CALMRegressor,
}
