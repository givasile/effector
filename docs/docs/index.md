# Home

`Effector` is an explainability package for tabular data.
It offers a variety of [global](./global_effect_intro/) and [regional](./regional_effect_intro/) effect methods, under a unified API.

---

`Effector` is compatible with `Python 3.7+`. 
Install `Effector` via `pip`:

```bash
pip install effector
```

---

- Start using `Effector` with the [Quickstart](./quickstart/).
- Learn more on how to use `Effector` in the [API DOCS](./api_docs/).
- Learn more about it through our [Guides](./guides/).
- Check out the [Examples](./examples/) to see `Effector` in action.

--- 

If you find `Effector` useful in your research, please consider citing the following papers:

```bibtex
@misc{gkolemis2024effector,
      title={Effector: A Python package for regional explanations}, 
      author={Vasilis Gkolemis and Christos Diou and Eirini Ntoutsi and Theodore Dalamagas and Bernd Bischl and Julia Herbinger and Giuseppe Casalicchio},
      year={2024},
      eprint={2404.02629},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

- [Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.](https://projecteuclid.org/euclid.aos/1013203451)
- [Apley, Daniel W. "Visualizing the effects of predictor variables in black box supervised learning models." arXiv preprint arXiv:1612.08468 (2016).](https://arxiv.org/abs/1612.08468)
- [Gkolemis, Vasilis, "RHALE: Robust and Heterogeneity-Aware Accumulated Local Effects"](https://ebooks.iospress.nl/doi/10.3233/FAIA230354)
- [Gkolemis, Vasilis, "DALE: Decomposing Global Feature Effects Based on Feature Interactions"](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)
- [Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." Advances in neural information processing systems. 2017.](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [REPID: Regional Effect Plots with implicit Interaction Detection](https://proceedings.mlr.press/v151/herbinger22a.html)
- [Decomposing Global Feature Effects Based on Feature Interactions](https://arxiv.org/pdf/2306.00541.pdf)
- [Regionally Additive Models: Explainable-by-design models minimizing feature interactions](https://arxiv.org/abs/2309.12215)

---

`Effector` implements the following methods:

|  Method  |                      Global Effect                      |                                 Regional Effect                                 |                                                                       Paper                                                                        |                                                                                                                                
|:--------:|:-------------------------------------------------------:|:-------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
|   PDP    |     [`PDP`](./api/#effector.global_effect_pdp.PDP)      |        [`RegionalPDP`](./api/#effector.regional_effect_pdp.RegionalPDP)         | [PDP](https://projecteuclid.org/euclid.aos/1013203451), [ICE](https://arxiv.org/abs/1309.6392), [GAGDET-PD](https://arxiv.org/pdf/2306.00541.pdf)  |
|  d-PDP   |  [`DerPDP`](./api/#effector.global_effect_pdp.DerPDP)   |     [`RegionalDerPDP`](./api/#effector.regional_effect_pdp.RegionalDerPDP)      |                                                  [d-PDP, d-ICE](https://arxiv.org/abs/1309.6392)                                                   | 
|   ALE    |     [`ALE`](./api/#effector.global_effect_ale.ALE)      |        [`RegionalALE`](./api/#effector.regional_effect_ale.RegionalALE)         |                [ALE](https://academic.oup.com/jrsssb/article/82/4/1059/7056085), [GAGDET-ALE](https://arxiv.org/pdf/2306.00541.pdf)                |                                                                                    
|  RHALE   |   [`RHALE`](./api/#effector.global_effect_ale.RHALE)    |      [`RegionalRHALE`](./api/#effector.regional_effect_ale.RegionalRHALE)       |         [RHALE](https://ebooks.iospress.nl/doi/10.3233/FAIA230354), [DALE](https://proceedings.mlr.press/v189/gkolemis23a/gkolemis23a.pdf)         |
| SHAP-DP  |  [`ShapDP`](./api/#effector.global_effect_shap.ShapDP)  |     [`RegionalShapDP`](./api/#effector.regional_effect_shap.RegionalShapDP)     | [SHAP](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions), [GAGDET-DP](https://arxiv.org/pdf/2306.00541.pdf)  |

