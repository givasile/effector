# Methods

This section explains how each method (PDP, ALE, RHALE) defines and estimates the global effects and the heterogeneity.

## Notation

Let \( x_s \) represent the feature of interest, and \( \mathbf{x}_c \) represent the other features.

1. **Global Effect Definition**:  
   Using method \( m \), the global effect of \( x_s \) is a $1$-dimensional function denoted as \( f^m(x_s) \). 
   
2. **Centered Global Effect Definition**:
   If the global effect is centered, it is denoted as \( f_c^m(x_s) \), where:
    $$
    f_c^m(x_s) = f^m(x_s) - c, \text{ where } c = \frac{\int_{x_{s,\text{min}}}^{x_{s,\text{max}}} f^m(x_s) \, dx_s}{x_{s,\text{max}} - x_{s,\text{min}}}
    $$
    The normalizer \( c \) is the mean value of the global effect over the range of \( x_s \).

3. **Heterogeneity Definition**:  
   Using method \( m \), the heterogeneity of the effect of \( x_s \) is a $1$-dimensional positive function denoted as \( H^m(x_s) \).
   The heterogeneity as a value is the mean of the heterogeneity over the range of \( x_s \):
   $$
   H_{x_s}^m = \frac{\int_{x_{s,\text{min}}}^{x_{s,\text{max}}} H^m(x_s) \, dx_s}{x_{s,\text{max}} - x_{s,\text{min}}}.
   $$

3. **Global Effect Estimation**:
    The estimation of the global effect of \( x_s \) using method \( m \) is denoted as \( \hat{f}^m(x_s) \).

4. **Centered Global Effect Estimation**:
   The centered global effect estimation is denoted as \( \hat{f}_c^m(x_s) \), where:

    $$
    \hat{f}_c^m(x_s) = \hat{f}^m(x_s) - c
    $$

    The normalizer \( c \) is the mean value of the global effect over the range of \( x_s \):
    $$
    c = \frac{\Delta x}{K} \sum_{k=1}^{K} \hat{f}^m(x_{s,\text{min}} + k \Delta x) \text{, where } \Delta x = \frac{x_{s,\text{max}} - x_{s,\text{min}}}{K}
    $$

5. **Heterogeneity Estimation**:
    The estimation of the heterogeneity of the effect of \( x_s \) using method \( m \) is denoted as \( \hat{H}^m(x_s) \).

`Effector` uses the estimations to compute the global effect and the heterogeneity of the effect of a feature of interest.

## Partial Dependence Plot (PDP)

### Global Effect

#### Definition

\begin{equation}
f^{PDP}(x_s) = \mathbb{E}_{\mathbf{X}_c} [f(x_s, \mathbf{X}_c)]
\end{equation}

#### Estimation
\begin{equation}
\hat{f}^{PDP}(x_s) = \frac{1}{N} \sum_{i=1}^{N} f(x_s, x_c^i))
\end{equation}

### Heterogeneity 

#### Definition

\begin{equation}
H^{PDP}(x_s) = E_{\mathbf{X}_c} [ \left ( f(x_s, \mathbf{X}_c) - f^{PDP}(x_s) \right )^2 ]
\end{equation}

\begin{equation}
H^{PDP}_{x_s} = \frac{\int_{x_{s,min}}^{x_{s,max}} H^{PDP}(x_s) dx_s}{x_{s,max} - x_{s,min}}
\end{equation}

#### Estimation

\begin{equation}
H^{PDP}(x_s) = \frac{1}{N} \sum_{i=1}^{N} \left ( f(x_s, x_c^i) - \hat{f}^{PDP}(x_s) \right )^2
\end{equation}

\begin{equation}
\hat{H}^{PDP}_{x_s} = \frac{\Delta x}{K} \sum_{k=1}^{K} \hat{H}^{PDP}(x_{s,min} + k \Delta x)
\end{equation}

