# Effector

[Documenation](https://xai-effector.github.io/) | [Global Effect](https://xai-effector.github.io/Feature%20Effect/01_global_effect_intro/) | [Regional Effect](https://xai-effector.github.io/Feature%20Effect/02_regional_effect_intro/) | [API](https://xai-effector.github.io/api/) | [Tutorials](https://xai-effector.github.io/)

`Effector` is a python package for global and regional effect analysis.

---

![using effector](docs/docs/static/effector_intro.gif)

---
### Installation

`Effector` is compatible with `Python 3.7+`. We recommend to first create a virtual environment with `conda`:

```bash
conda create -n effector python=3.11
conda activate effector
```

and then install `Effector` via `pip`:

```bash
pip install effector
```

If you want to also run the Tutorial notebooks, add some more dependencies to the environment:

```bash
pip install -r requirements-dev.txt
```

## Methods and Publications

### Methods

`Effector` implements the following methods:

| Method   | Global Effect                                             | Regional Effect                                                               | Paper                                                                                                                                               |                                                                                                                                
|----------|-----------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| PDP      | [`PDP`](./api/#effector.global_effect_pdp.PDP)            | [`RegionalPDP`](./api/#effector.regional_effect_pdp.RegionalPDP)              | [PDP](https://projecteuclid.org/euclid.aos/1013203451), [ICE](https://arxiv.org/abs/1309.6392), [GAGDET-PD](https://arxiv.org/pdf/2306.00541.pdf)   |
| d-PDP    | [`DerPDP`](./api/#effector.global_effect_pdp.DerPDP)      | [`RegionalDerPDP`](./api/#effector.regional_effect_pdp.RegionalDerPDP)        | [d-PDP, d-ICE](https://arxiv.org/abs/1309.6392)                                                                                                     | 
| ALE      | [`ALE`](./api/#effector.global_effect_ale.ALE)            | [`RegionalALE`](./api/#effector.regional_effect_ale.RegionalALE)              | [ALE](https://academic.oup.com/jrsssb/article/82/4/1059/7056085), [GAGDET-ALE](https://arxiv.org/pdf/2306.00541.pdf)                                |                                                                                    
| RHALE    | [`RHALE`](./api/#effector.global_effect_ale.RHALE)        | [`RegionalRHALE`](./api/#effector.regional_effect_ale.RegionalRHALE)          | [RHALE](https://ebooks.iospress.nl/doi/10.3233/FAIA230354), [DALE](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)                  |
| SHAP-DP  | [`ShapDP`](./api/#effector.global_effect_shap.ShapDP)     | [`RegionalShapDP`](./api/#effector.regional_effect_shap.RegionalShapDP)       | [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions), [GAGDET-DP](https://arxiv.org/pdf/2306.00541.pdf)   |
