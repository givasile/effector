# Methods

This section explains how each method (PDP, ALE, RHALE) defines and estimates the global effects and the heterogeneity.

## Notation

- \( x_s \) is the feature of interest and \( \mathbf{x}_c \) the rest, so that \( \mathbf{x} = (x_s, \mathbf{x}_c) \).
- \( f^m(x_s) \) or \( m(x_s) \) is the global effect of feature \( x_s \) using method \( m \), for example $f^{PDP}(x_s)$ or $PDP(x_s)$.
- $f_c^m(x_s)$ or $m_c(x_s)$ is the centered global effect:
    $$
    f_c^m(x_s) = f^m(x_s) - c, \text{ where } c = \frac{\int_{x_{s,\text{min}}}^{x_{s,\text{max}}} f^m(x_s) \, dx_s}{x_{s,\text{max}} - x_{s,\text{min}}}
    $$
    The normalizer \( c \) is the mean value of the global effect over the range of \( x_s \).
- \( h^m(x_s) \) is the *heterogeneity function* of the effect of \( x_s \) using method \( m \).
- \( H^m_{x_s} \) is the *heterogeneity value* of the effect of \( x_s \) using method \( m \):
    $$
    H_{x_s}^m = \frac{\int_{x_{s,\text{min}}}^{x_{s,\text{max}}} h^m(x_s) \, dx_s}{x_{s,\text{max}} - x_{s,\text{min}}}.
    $$

## PDP

### Global effect

The Partial Dependence Plot (PDP) is defined as the average of the predictions over the rest of the features:

\begin{equation}
PDP(x_s) = \frac{1}{N} \sum_{i=1}^{N} f(x_s, \mathbf{x}_c^i))
\end{equation}

and the centered global effect is $PDP_c(x_s) = PDP(x_s) - c$.

### Heterogeneity

The ICE plot is the effect of the feature \( x_s \) for each instance \( i \):

\begin{equation}
ICE^i(x_s) = f(x_s, \mathbf{x}_c^i)
\end{equation}

The heterogeneity function is the mean squared difference between the centered-ICE plot and the centered-PDP plot:

\begin{equation}
h^{PDP}(x_s) = \frac{1}{N} \sum_{i=1}^{N} \left ( ICE_c^i(x_s) - PDP_c(x_s) \right )^2
\end{equation}

and the heterogeneity value is \( H_{x_s}^{PDP} \).

## ALE

### Global effect

ALE, first, partitions the range \( [x_{s,\text{min}}, x_{s,\text{max}}] \) into \( K \) intervals (bins),
each containing \( \mathcal{S}_k \) instances (the ones with \( x_s \) in the \( k \)-th bin).

Each instance has a (local) effect that is computed as the difference between 
the prediction of the model after setting \(x_s\) to the right and the left boundary of the bin:

\begin{equation}
\Delta f^i = f(z_k, x^i_2, x^i_3) - f(z_{k-1}, x^i_2, x^i_3)
\end{equation}

On bin $k$, the bin-effect is the average of the local effects:

\begin{equation}
\mu_k = \frac{1}{| \mathcal{S}_k |} \sum_{i: x^i \in \mathcal{S}_k} \left [  \Delta f^i \right ]
\end{equation}

and the ALE effect at $x_s$ is the sum of the bin-effects up to the bin containing $x_s$:

\begin{equation}
ALE(x_s) = \sum_{k=1}^{k_{x_s}} \mu_k
\end{equation}

The centered global effect is \( ALE_c(x_s) = ALE(x_s) - c \).

### Heterogeneity

The bin-variance, $\text{Var}_k$, is the variance of the local effects in the bin $k$:

\begin{equation}
\text{Var}_k = \frac{1}{| \mathcal{S}_k |} \sum_{i: x^i \in \mathcal{S}_k} \left [  \Delta f^i - \mu_k \right ]^2
\end{equation}


The heterogeneity function $h^{ALE}(x_s)$ equals to the bin-variance of the bin containing $x_s$:

\begin{equation}
h^{ALE}(x_s) = \text{Var}_{k(x_s)}
\end{equation}

The heterogeneity value is the mean of these variances:

\begin{equation}
H_{x_s}^{ALE} = \frac{1}{K} \sum_{k=1}^{K} \text{Var}_k
\end{equation}