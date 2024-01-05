# Intro to Feature Effect methods with a linear model

This tutorial is a (slow and gentle) introduction to the basic global [feature effect methods](https://christophm.github.io/interpretable-ml-book/global-methods.html) and the `Effector` package. If you only care about using `Effector`'s API, you can go directly to the [Conclusion](#conclusion).



```python
import numpy as np
import effector
```

## Global Feature Effect methods

Feature effect methods estimate the **average** effect of a specific feature on the model's output, i.e., they create a 1-1 mapping between the feature of interest $x_s$ and the output of the model $y$. 

Feature effect are perfect explainers for additive models, i.e., models of the form $f(\mathbf{x}) = \sum_{i=1}^D f(x_i)$. Black-box models, however, are not additive; they have complex interaction terms between two $f(x_i, x_j)$, three $f(x_i, x_j, x_k)$ or even all features $f(x_1, \cdots, x_D)$. In these cases, feature effect methods simplify things by _distributing_ the effect of interaction terms to the individual features. 

This simplification is acceptable when the interaction terms are weak, i.e., they are not so important for the model's prediction.
However, when the interaction terms are very strong then the feature effect methods may provide an over-simplistic explanation.
Hopefully, there is a quantity called **heterogeneity** that can be used to check whether the feature effect methods are a good explanation for the features.

 `Effector` provides five different feature effect methods, which are summarized in the table below. In all methods, setting `heterogeneity=True` the methods show the level of heterogeneity, along with the average effect.

<center>

| Method                                  | Description                        | API in `Effector`                                                            | Paper                                                             |
|-----------------------------------------|------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [PDP](#partial-dependence-plot-pdp)     | Partial Dependence Plot            | [PDP]((./../../reference/#effector.pdp.DerivativePDP))                       | [Friedman, 2001](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) |     |
| [d-PDP](#derivative-pdp-d-pdp)          | Derivative PDP                     | [DerivativePDP](./../../reference/#effector.pdp.DerivativePDP)               | [Goldstein et. al, 2013](https://arxiv.org/pdf/1309.6392.pdf)     ||
| [ALE](#accumulated-local-effects-ale)   | Accumulated Local Effect           | [ALE](./../../reference/#effector.ale.ALE)                                   | [Apley et. al, 2016](https://arxiv.org/pdf/1612.08468)            |
| [RHALE](#robust-and-heterogeneity-aware-ale-rhale) | Robust and Heterogeneity-aware ALE | [RHALE](./../../reference/#effector.ale.RHALE)                               | [Gkolemis et al, 2023](https://arxiv.org/abs/2309.11193)          | 
| [SHAP](#shap-dependence-plot)           | SHAP Dependence Plot               | [SHAPDependence](./../../reference/#effector.shap_dependence.SHAPDependence) | [Lundberg et. al, 2017](https://arxiv.org/pdf/1705.07874.pdf)     |


</center>

For the rest of the tutorial, we will use the following notation for the rest of the tutorial:

<center>

| Symbol                                                     | Description                                             |
|------------------------------------------------------------|---------------------------------------------------------|
| $f(\mathbf{x})$                                            | The black box model                                     |
| $\mathcal{H}$                                              | The heterogeneity                                       |
| $x_s$                                                      | The feature of interest                                 |
| $x_c$                                                      | The remaining features, i.e., $\mathbf{x} = (x_s, x_c)$ |
| $\mathbf{x} = (x_s, x_c) = (x_1, x_2, ..., x_s, ..., x_D)$ | The input features                                      |
| $\mathbf{x}^{(i)} = (x_s^{(i)}, x_c^{(i)})$                | The $i$-th instance of the dataset                      |

</center>

---
## Dataset and Model

In this example, we will use as black-box function, a simple linear model $y = 7x_1 - 3x_2 + 4x_3$. Since there are no interactions 
terms we expect **all** methods to provide the following feature effects and zero heterogeneity:

<center>

| Feature | Feature Effect | Heterogeneity |
| --- | --- | --- |
| $x_1$ | $7x_1$ | 0 |
| $x_2$ | $-3x_2$ | 0 |
| $x_3$ | $4x_3$ | 0 |

</center>


As dataset, we will generate $N=1000$ examples comming from the following distribution:

| Feature | Description                                                      | Distribution                      |
|---------|------------------------------------------------------------------|-----------------------------------|
| $x_1$   | Uniformly distributed between $0$ and $1$                        | $x_1 \sim \mathcal{U}(0,1)$       |
| $x_2$   | Follows $x_1$ with some added noise                              | $x_2 = x_1 + \epsilon $, $\epsilon \sim \mathcal{N}(0, 0.1)$ |
| $x_3$   | Uniformly distributed between $0$ and $1$                        | $x_3 \sim \mathcal{U}(0,1)$       |


```python
def generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma):
    x1 = np.random.uniform(x1_min, x1_max, size=int(N))
    x2 = np.random.normal(loc=x1, scale=x2_sigma)
    x3 = np.random.uniform(x1_min, x1_max, size=int(N))
    return np.stack((x1, x2, x3), axis=-1)

# generate the dataset
np.random.seed(21)

N = 1000
x1_min = 0
x1_max = 1
x2_sigma = .1
x3_sigma = 1.
X = generate_dataset(N, x1_min, x1_max, x2_sigma, x3_sigma)
```


```python
def predict(x):
    y = 7*x[:, 0] - 3*x[:, 1] + 4*x[:, 2]
    return y

def predict_grad(x):
    df_dx1 = 7 * np.ones([x.shape[0]])
    df_dx2 = -3 * np.ones([x.shape[0]])
    df_dx3 = 4 * np.ones([x.shape[0]])
    return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)
```



---
## Partial Dependence Plot (PDP)

The PDP is defined as **_the average prediction over the entire dataset when setting the feature of interest at a specific value._**
For example, the effect of the $s$-th feature at values $x_s$ is defined as:

$$ \text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)] $$ 

and is approximated by 

$$ \hat{\text{PDP}}(x_s) = \frac{1}{N} \sum_{j=1}^N f(x_s, x^{(i)}_c) $$

In practice, for all the dataset instances, we set the feature of interest at a specific value $x_s$ and we average the model's predictions.
Let's check it out the PDP effect using `effector`.


```python
effector.PDP(data=X, model=predict).plot(feature=0, show_avg_output=True, y_limits=[0,10])
effector.PDP(data=X, model=predict).plot(feature=1, show_avg_output=True, y_limits=[0,10])
effector.PDP(data=X, model=predict).plot(feature=2, show_avg_output=True, y_limits=[0,10])
```


    
![png](01_linear_model_files/01_linear_model_8_0.png)
    



    
![png](01_linear_model_files/01_linear_model_8_1.png)
    



    
![png](01_linear_model_files/01_linear_model_8_2.png)
    


### Feature effect interpretation

As we expected, all feature effects are linear. Looking closer, we can also confirm the gradients of the effects: 
$7$ for $x_1$, $-3$ for $x_2$ and $4$ for $x_3$.
However, you may question why there are different alignments on the y axis? For example, why $x_1$ starts at $y=-1.5$? Does this have a natural interpretation?

There is no global answer to this question. This is why many people prefer to _center_ PDP plots manually, as we will see below.
For linear models, the PDP plot is $\text{PDP}(x_s) = a_sx_s + c$ where $a_s$ is the gradient of the line and $c$ is the intercept. 
For feature $x_1$, the intercept is $c \approx 0.5$. 
With a closer look at the formula we can understand why this happens:

$$PDP(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)] = a_sx_s + \sum_{j \neq s} a_j \mathbb{E}_{x_j}[x_j] = a_sx_s - 3 * 0.5 + 4 * 0.5 = a_sx_s + 0.5$$

---

The most convenient centering of the PDP plot depends on the underlying question.
If we compare the effect of two features, then it is better to center the PDP plots around $y=0$ to avoid the distraction of intercepts.
If we compare the effect of a specific feature on two different subgroups (check the tutorial about Regional Effect methods), then it is better to to leave the PDP plot uncentered.

`Effector` has three `cenetering` alternatives:

| `centering`               | Description                            | Formula                                                                 |
|---------------------------|----------------------------------------|-------------------------------------------------------------------------|
| `False`                   | Don't enforce any additional centering | -                                                                       |
| `True` or `zero_integral` | Center around the $y$ axis             | $c = \mathbb{E}_{x_s \sim \mathcal{U(x_{s,min},x_{s, max})}}[PDP(x_s)]$ |
| `zero_start`              | Center around $y=0$                    | $c = 0$                                                                 |

Below, we observe that setting `centering=True` facilitates the comparisons. 


```python
effector.PDP(data=X, model=predict).plot(feature=0, centering=True, y_limits=[-4, 4])
effector.PDP(data=X, model=predict).plot(feature=1, centering=True, y_limits=[-4, 4])
effector.PDP(data=X, model=predict).plot(feature=2, centering=True, y_limits=[-4, 4])
```


    
![png](01_linear_model_files/01_linear_model_10_0.png)
    



    
![png](01_linear_model_files/01_linear_model_10_1.png)
    



    
![png](01_linear_model_files/01_linear_model_10_2.png)
    


### Heterogeneity

Feature effect methods output a 1-1 plot that visualizes the **average** effect of a specific feature on the output; 
the averaging is performed over the instance-level effects. 
It is important, therefore, to know to what extent the underlying local (instance-level) effects deviate from the average effect.
In other words, to what extent the average effects are a good explanation for the features.

In our example, due to zero interactions between the features, the heterogeneity should be zero.

---
In PDP plots there are two ways to check that, either using the ICE plots or as a $\pm$ interval around the average plot.

#### Option (a): ICE plots

ICE plots show the ouput of instance $i$ if changing the feature of interest $x_s$:

$$\text{ICE}^{(i)}(x_s, x^{(i)}_c) = f(x_s, x^{(i)}_c)$$

Plotting the ICE plots of many instances $i$ on top of the PDP, we can visually observe the heterogeneity.
For example in the plot below, we can see that there is no heterogeneity in the instance-level effects, i.e., all instance-level effects are lines with gradient 7.



```python
effector.PDP(data=X, model=predict).plot(feature=0, centering=True, heterogeneity="ice")
```


    
![png](01_linear_model_files/01_linear_model_13_0.png)
    


Keep in mind that it is important to define correctly the argument `centering`.
Setting `centering=True` centers the PDP plot around the $y$ axis, which facilitates the comparison of the underlying feature effects (as above).

However, there are cases where the intercept maybe useful.
Imagine a case where the salary of the employees depends only (a) on the number of working hours and (b) on the gender of a person. 
If the salary per working hour does not differ between male and female employees, but male employees in general earn 1000 Euros more per month, 
then we won’t see this difference in the centered ICE curves of the feature working hours.
In contrast, this difference will be visible under the uncentered ICE curves.

In our example, setting `centering=False` gives the following plot; ICE plots with different intercepts but identical gradient.


```python
effector.PDP(data=X, model=predict).plot(feature=0, centering=False, heterogeneity="ice")
```


    
![png](01_linear_model_files/01_linear_model_15_0.png)
    


#### Option (b): STD of the residuals

A second way to check for heterogeneity is by plotting the standard deviation of the instance-level effects as $\pm$ interval around the PDP plot.
This is done setting `confidence_interval="std"` in the `plot` method. 
In practice, this approach simply plots the std of the ICE plots instead of the ICE plots themselves.



```python
effector.PDP(data=X, model=predict).plot(feature=0, centering=True, heterogeneity="std")
```


    
![png](01_linear_model_files/01_linear_model_17_0.png)
    


In this case, if we do not perform centering, it is difficult to differentiate whether the heterogeneity is provoked by the gradient or in the intercept.
Therefore, we recommend to try both the centered and the uncentered version of the ICE, before coming to a conclusion. 
If the heterogeneity is only present on the latter, then it is due to different intercepts.


```python
effector.PDP(data=X, model=predict).plot(feature=0, centering=False, heterogeneity="std")
```


    
![png](01_linear_model_files/01_linear_model_19_0.png)
    


### Derivative-PDP (d-PDP)

A similar analysis can be done using the derivative of the model; the name of this approach is Derivative-PDP (d-PDP) and the equivalent of the ICE plots are the Derivative-ICE (d-ICE) plots. The d-PDP and d-ICE are defined as:

$$ \text{d-PDP}(x_s) = \mathbb{E}_{x_c}[\frac{\partial f}{\partial x_s} (x_s, x_c)] \approx \frac{1}{N} \sum_{j=1}^N \frac{\partial f}{\partial x_s} (x_s, x_c^{(i)}) $$

and 

$$ \text{d-ICE}^{(i)}(x_s) = \frac{\partial f}{\partial x_s} (x_s, x^{(i)}_c) $$

We have to mention that:
 
* d-PDP needs the model's gradient, which is not always available.
* Under normal circumstances, the d-PDP and d-ICE should not be centered because the absolute value of the derivative has a natural meaning for the interpretation. In practice, d-ICE plots show variation that is only due to difference in the shapes of the curves. This is because all terms that are not related (interact) with the feature of interest will become zero when taking the derivative. The same applies for the $\pm$ interval around the d-PDP plot.
* The interpretation is given in the gradient-space, so it should be treated differently. In d-PDP the plots show how much the model's prediction *changes* given a change in the feature of interest. This is different from PDP, where the plots says how much the specific feature *contributes* to the prediction. 
* d-PDP is the gradient of the PDP, i.e., $\text{d-PDP}(x) = \frac{\partial \text{PDP}}{\partial x_s} (x)$
* d-ICE is the gradient of the ICE, i.e., $\text{d-ICE}^{(i)}(x) = \frac{\partial \text{ICE}^{(i)}}{\partial x_s} (x)$

As we can see below, the standard deviation of the ICE plots is zero, because they only measure the variation of the shapes of the curves; not the variation of the intercepts.


```python
effector.DerivativePDP(data=X, model=predict, model_jac=predict_grad).plot(feature=0, heterogeneity=True)
effector.DerivativePDP(data=X, model=predict, model_jac=predict_grad).plot(feature=0, heterogeneity="ice")
```


    
![png](01_linear_model_files/01_linear_model_21_0.png)
    



    
![png](01_linear_model_files/01_linear_model_21_1.png)
    


## Accumulated Local Effects (ALE)

The next major category of feature effect techniques is [Accumulated Local Effects (ALE)](https://christophm.github.io/interpretable-ml-book/ale.html). Before we go into the specifics, let's apply the ALE plot to our example.


```python
effector.ALE(data=X, model=predict).plot(feature=0)
effector.ALE(data=X, model=predict).plot(feature=1)
effector.ALE(data=X, model=predict).plot(feature=2)
```


    
![png](01_linear_model_files/01_linear_model_23_0.png)
    



    
![png](01_linear_model_files/01_linear_model_23_1.png)
    



    
![png](01_linear_model_files/01_linear_model_23_2.png)
    


### Fearure effect interpretation

In each of the figures above, there are two subfigures; the upper subfigure is the average feature effect (the typical ALE plot) and the lower subfigure is the derivative of the effect.
The upper subfigure shows how much the feature of interest _contributes_ to the prediction (like PDP) while the bottom subplot shows how much a change in the feature of interest _changes_ the prediction (like d-PDP). 
For example, for $x_1$ the upper subplot shows a linear effect and the lower subplot confirms that the gradient is constantly $7$.
`Effector` offers two alternatives for centering the ALE plot.

<center>

| `centering`               | Description                            | Formula                                                               |
|---------------------------|----------------------------------------|-----------------------------------------------------------------------|
| `False` or `zero_start`   | Don't enforce any additional centering | c=0                                                                   |
| `True` or `zero_integral` | Center around the $y$ axis             | c=$\mathbb{E}_{x_s \sim \mathcal{U(x_{s,min},x_{s, max})}}[ALE(x_s)]$ |

</center>
Let's see how centering works for $x_1$:


```python
effector.ALE(data=X, model=predict).plot(feature=0, centering=True)
```


    
![png](01_linear_model_files/01_linear_model_25_0.png)
    


### Heterogeneity 

In ALE plots, the only way to check the heterogeneity of the instance-level effects is by plotting the standard deviation of the instance-level effects as $\pm$ interval around the ALE plot. In `Effector` this can be done by setting `heterogeneity=True"`. The plot below shows that the heterogeneity is zero, which is correct. However, as we will see below [(RHALE section)](#robust-and-heterogeneity-aware-ale-rhale), ALE's fixed size bin-splitting is not the best way to estimate the heterogeneity. In contrast, the automatic bin-splitting introduced by [RHALE](https://arxiv.org/abs/2309.11193) provides a better estimation of the heterogeneity.


```python
effector.ALE(data=X, model=predict).plot(feature=0, centering=True, heterogeneity=True)
```


    
![png](01_linear_model_files/01_linear_model_27_0.png)
    


### Bin-Splitting

As you may have noticed at the bottom plots of the figures above, $x_1$ axis has been split in $K=20$ bins (intervals) of equal size. The derivative-effect is provided per bin (bin-effect), which in our example is $7$ for all bins. 

In fact, bin-splitting is apparent also at the top plot; the top plot is not a line, but a piecewise linear function, where each _piece_ is a line in the are covered by each bin and gradient equal to the bin-effect. However, since the bin-effect is the same for all bins, the top plot looks like a line.

To explain the need for bin-splitting we have to go back to the definition of ALE. ALE is defined as: 

$$\text{ALE}(x_s) = \int_{z=0}^{x_s} \mathbb{E}_{x_c|x_s=z}\left [ \frac{\partial f}{\partial x_s} (z, x_c) \right ] \partial z$$

Apley et. al proposed approximating the above integral by:

$$\hat{\text{ALE}}(x_s) = \sum_{k=1}^{k_{x_s}} \frac{1}{| \mathcal{S}_k |} \sum_{i: x^{(i)} \in \mathcal{S}_k} \left [ f(z_k, x_c) - f(z_{k-1}, x_c) \right ]$$

where $k_{x_s}$ the index of the bin such that $z_{k_{x−1}} ≤ x_s < z_{k_x}$, $\mathcal{S}_k$ is the set of the instances lying at the $k$-th bin, i.e., $\mathcal{S}_k = \{ x^{(i)} : z_{k−1} \neq x^{(i)}_s < z_k \}$ and $\Delta x = \frac{x_{s, max} - x_{s, min}}{K}$.

$\hat{\text{ALE}}(x_s)$ uses a Riemannian sum to approximate the integral of $\text{ALE}(x_s)$. The axis of the $s$-th feature is split in $K$ bins (intervals) of equal size. In each bin, the average effect of the feature of interest is estimated using the instances that fall in the bin. The average effect in each bin is called bin-effect. The default in `Effector` is to use $K=20$ bins but the user can change it using:


```python
ale = effector.ALE(data=X, model=predict)

# using 5 bins
bm = effector.binning_methods.Fixed(nof_bins=5, min_points_per_bin=0, cat_limit=10)
ale.fit(features=0, binning_method=bm)
ale.plot(feature=0)

# using 100 bins
bm = effector.binning_methods.Fixed(nof_bins=100, min_points_per_bin=0, cat_limit=10)
ale.fit(features=0, binning_method=bm)
ale.plot(feature=0)
```


    
![png](01_linear_model_files/01_linear_model_29_0.png)
    



    
![png](01_linear_model_files/01_linear_model_29_1.png)
    


## Robust and Heterogeneity-aware ALE (RHALE)

Robust and Heterogeneity-aware ALE (RHALE) is a variant of ALE, proposed by [Gkolemis et. al](https://arxiv.org/abs/2309.11193). In their paper, they showed that RHALE has specific advantages over ALE: (a) it ensures on-distribution sampling (b) an unbiased estimation of the heterogeneity and (c) an optimal trade-off between bias and variance. These are achieved using an automated variable-size binning splitting approach. Let's see how it works in practice.


```python
effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(feature=0, centering=False, show_avg_output=False)
```


    
![png](01_linear_model_files/01_linear_model_31_0.png)
    


### Fearure effect interpretation

The interpretation is exactly the same as with the typical ALE; The top subplot is the average feature effect and the bottom subfigure is the derivative of the effect. 
The crucial difference, is that the automatic bin-splitting approach _optimally_ creates a single bin that covers the whole area between $x=0$ and $x=1$. As we saw above, the gradient of the feature effect is constant and equal to $7$ for all $x_1$ values. Therefore, merging all bins into one, reduces the variance of the estimation; the estimation is based on more instances, so the variance is lower. 

In our example, this advantage is not evident; Since there are no interaction terms (linear model) the effect of all instances is always the same; so the variance of the estimation is zero. However, in more complex models, the variance of the estimation is not zero and the automatic bin-splitting approach reduces the variance of the estimation (check tutorial [ALE](./ale.ipynb) for more details).

As with the ALE, there are two alternatives for centering the ALE plot.

<center>

| `centering`               | Description                            | Formula                                                               |
|---------------------------|----------------------------------------|-----------------------------------------------------------------------|
| `False` or `zero_start`   | Don't enforce any additional centering | c=0                                                                   |
| `True` or `zero-integral` | Center around the $y$ axis             | c=$\mathbb{E}_{x_s \sim \mathcal{U(x_{s,min},x_{s, max})}}[ALE(x_s)]$ |

</center>

Let's see how this works for $x_1$:


```python
effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(feature=0, centering=True, show_avg_output=False)
```


    
![png](01_linear_model_files/01_linear_model_33_0.png)
    


### Heterogeneity

As before, the heterogeneity is given by the the standard deviation of the instance-level effects as $\pm$ interval around the ALE plot.
It is important to notice, that automatic bin-splitting provides a better estimation of the heterogeneity, compared to the equisized binning method used by ALE. (check tutorial [ALE](./ale.ipynb) for more details). 
The plot below correctly informs shows that the heterogeneity is zero.


```python
effector.RHALE(data=X, model=predict, model_jac=predict_grad).plot(feature=0, centering=True, heterogeneity="std", show_avg_output=False)
```


    
![png](01_linear_model_files/01_linear_model_35_0.png)
    


### Bin-Splitting

So how the automatic bin-splitting works? 

$$\text{ALE}(x_s) = \int_{z=0}^{x_s} \mathbb{E}_{x_c|x_s=z}\left [ \frac{\partial f}{\partial x_s} (z, x_c) \right ] \partial z$$

and is approximated by:

$$\hat{\text{RHALE}}(x_s) = \sum_{k=1}^{k_{x_s}} \frac{1}{ \left | \mathcal{S}_k \right |} \sum_{i: x^{(i)} \in \mathcal{S}_k} \frac{\partial f}{\partial x_s} (x_s^{(i)}, x_c^{(i)})$$


The above approximation uses a Riemannian sum to approximate the integral. The axis of the $s$-th feature is split in $K$ bins (intervals) of equal size. In each bin, the average effect of the feature of interest is estimated using the instances that fall in the bin. The average effect in each bin is called bin-effect. 

But what we saw above is different. In the figure above, only one bin has been created and covers the whole area between $x=0$ and $x=1$. 
This is because the default behaviour of `Effector` is to use an automatic bin-splitting method, as it was proposed by [Gkolemis et. al](https://arxiv.org/abs/2309.11193).
For more details about that, you can check the in-depth [ALE tutorial](./ale.ipynb).

## SHAP Dependence Plot

TODO add intro


```python
effector.SHAPDependence(data=X, model=predict).plot(feature=0, centering=False, show_avg_output=False)
```


    
![png](01_linear_model_files/01_linear_model_38_0.png)
    


### Fearure effect interpretation

TODO add content 

<center>

| `centering`               | Description                            | Formula                                                               |
|---------------------------|----------------------------------------|-----------------------------------------------------------------------|
| `False` or `zero_start`   | Don't enforce any additional centering | c=0                                                                   |
| `True` or `zero-integral` | Center around the $y$ axis             | c=$\mathbb{E}_{x_s \sim \mathcal{U(x_{s,min},x_{s, max})}}[ALE(x_s)]$ |

</center>

Let's see how this works for $x_1$:


```python
effector.SHAPDependence(data=X, model=predict).plot(feature=0, centering=True, show_avg_output=False)
```


    
![png](01_linear_model_files/01_linear_model_40_0.png)
    


### Heterogeneity

As before, the heterogeneity is given by the the standard deviation of the instance-level effects as $\pm$ interval around the ALE plot.
It is important to notice, that automatic bin-splitting provides a better estimation of the heterogeneity, compared to the equisized binning method used by ALE. (check tutorial [ALE](./ale.ipynb) for more details). 
The plot below correctly informs shows that the heterogeneity is zero.


```python
effector.SHAPDependence(data=X, model=predict).plot(feature=0, heterogeneity="shap_values")
effector.SHAPDependence(data=X, model=predict).plot(feature=0, heterogeneity="std")
```


    
![png](01_linear_model_files/01_linear_model_42_0.png)
    



    
![png](01_linear_model_files/01_linear_model_42_1.png)
    


## Conclusion

In this tutorial, we introduced the various feature effect methods of `Effector` and used them to explain a linear model. 

In summary, given a dataset `X: (N, D)` and a black-box model `model: (N, D) -> (N)`,
the feature effect plot of the $s$-th feature `feature=s` is given with the table below.
The argument `confidence_interval=True|False` indicates whether to plot the standard deviation of the instance-level effects as $\pm$ interval around the feature effect plot. Some methods also require the gradient of the model `model_jac: (N, D) -> (N, D)`.

<center>

| Method        | How to use                                                                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| PDP           | [`effector.PDP(X, model).plot(feature, centering, confidence_interval)`]((./../../reference/#effector.pdp.PDP))                    |
| d-PDP         | [`effector.DerivativePDP(X, model, model_jac).plot(feature, centering, confidence_interval)`](./../../reference/#effector.pdp.DerivativePDP) |
| ALE           | [`effector.ALE(X, model).plot(feature, centering, confidence_interval)`](./../../reference/#effector.ale.ALE)                                |
| RHALE         | [`effector.RHALE(X, model, model_jac).plot(feature, centering, confidence_interval)`](./../../reference/#effector.ale.RHALE)                 |

</center>


```python

```
