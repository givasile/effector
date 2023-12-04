# An introduction into Regional Feature Effects

This tutorial is a gentle introduction to Regional feature effects and the `Effector` package. Regional feature effects bridge the gap between local and global feature effects. The [REPID](https://proceedings.mlr.press/v151/herbinger22a/herbinger22a.pdf) method was recently introduced to find regions within the feature space which minimizes feature interactions (interaction-related heterogeneity) based on ICE curves and PDPs for one feature of interest. This approach was extended to ALE and SHAP Dependence Plots for multiple features of interest by [GADGET](https://arxiv.org/pdf/2306.00541.pdf) (with REPID being a special case of GADGET). Here, we illustrate the advantages of GADGET-PD and GADGET-ALE (RHALE) compared to their global alternatives in the presence of feature interactions. We also illustrate the advantage of GADGET-ALE compared to GADGET-PD in the presence of high correlations in the feature space.
GADGET-PD and GADGET-ALE are both provided in the `Effector` package. We showcase its userfriendly applicability in the following simulation example.

The tutorial is structured as follows:
 - Introduction of the simulation example: We consider a simple linear setting with subgroup-specific feature interactions, which will be defined with independent features and with dependent features.
 - Modeling: We fit a neural network on the two data sets (uncorrelated vs. correlated)
 - PDP - Influence of feature interactions and feature correlations
 - RHALE - Influence of feature interactions and feature correlations
 - Regional Effects: GADGET-PD and GADGET-ALE
 



```python
import numpy as np
import effector
import keras
import tensorflow as tf
```

---
## Introduction of the simulation example

We will generate $N=500$ examples with $D=3$ features, which are in the uncorrelated setting all uniformly distributed as follows:

<center>

| Feature | Description | Distribution                 |
| --- | --- |------------------------------|
| $x_1$ | Uniformly distributed between $-1$ and $1$ | $x_1 \sim \mathcal{U}(-1,1)$ |
| $x_2$ | Uniformly distributed between $-1$ and $1$ | $x_2 \sim \mathcal{U}(-1,1)$ |
| $x_3$ | Uniformly distributed between $-1$ and $1$ | $x_3 \sim \mathcal{U}(-1,1)$ |

</center>
For the correlated setting we keep the distributional assumptions for $x_2$ and $x_3$ but define $x_1$ such that it is highly correlated with $x_3$ by: $x_1 = x_3 + \delta$ with $\delta \sim \mathcal{N}(0,0.0625)$.


```python
def generate_dataset_uncorrelated(N):
    x1 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x3 = np.random.uniform(-1, 1, size=N)
    return np.stack((x1, x2, x3), axis=-1)

def generate_dataset_correlated(N):
    x3 = np.random.uniform(-1, 1, size=N)
    x2 = np.random.uniform(-1, 1, size=N)
    x1 = x3 + np.random.normal(loc = np.zeros_like(x3), scale = 0.25)
    return np.stack((x1, x2, x3), axis=-1)

# generate the dataset for the uncorrelated and correlated setting
np.random.seed(123)
N = 500
X_uncor = X_uncor_train = generate_dataset_uncorrelated(N)
X_uncor_test = generate_dataset_uncorrelated(10000)
X_cor = X_cor_train = generate_dataset_correlated(N)
X_cor_test = generate_dataset_correlated(10000)

```

---
## Modeling

We will use the following linear model with a subgroup-specific interaction term, $y = 3x_1I_{x_3>0} - 3x_1I_{x_3\leq0} + x_3 + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.09)$. On a global level, there is a high heterogeneity for the features $x_1$ and $x_3$ due to their interaction with each other. However, this heterogeneity vanishes to 0 if the feature space is separated into two regions with respect to $x_3 = 0$. In this case only main effects remain in the two regions: 

<center>

| Feature |Region | Average Effect | Heterogeneity |
| --- | --- | --- |--- |
| $x_1$ | $x_3>0$| $3x_1$ | 0 |
| $x_1$ | $x_3\leq 0$| $-3x_1$ | 0 |
| $x_3$ | $x_3>0$| $x_3$ | 0 |
| $x_3$ | $x_3\leq 0$| $x_3$ | 0 |

</center>

Since $x_2$ does not have any influence (neither main nor interaction effect) on the target, the average effect and the heterogeneity of this feature are $0$ (globally and regionally).
Note that the average effect of $x_1$ cancels out on a global level and thus only considering the average global effect would suggest no influence of the feature on the target.


```python
def generate_target(X):
    f = np.where(X[:,2] > 0, 3*X[:,0] + X[:,2], -3*X[:,0] + X[:,2])
    epsilon = np.random.normal(loc = np.zeros_like(X[:,0]), scale = 0.3)
    Y = f + epsilon
    return(Y)

# generate target for uncorrelated and correlated setting
np.random.seed(123)
Y_uncor_train = generate_target(X_uncor_train)
Y_uncor_test = generate_target(X_uncor_test)
Y_cor_train = generate_target(X_cor_train)
Y_cor_test = generate_target(X_cor_test)      
```

#### Fit a Neural Network

We train a single-layer feedforward Neural Network of size 10, a weight decay of 0.001 and a maximum number of iterations of 1000 (HPO was done by Herbinger et. al (2023) for this example) on the uncorrelated and the correlated setting.


```python
# Train - Evaluate - Explain a neural network
np.random.seed(123)

model_uncor = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=(3,)),
    keras.layers.Dense(10, activation="relu", input_shape=(3,)),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model_uncor.compile(optimizer=optimizer, loss="mse")
model_uncor.fit(X_uncor_train, Y_uncor_train, epochs=40)
model_uncor.evaluate(X_uncor_test, Y_uncor_test)
```

    Epoch 1/40
    16/16 [==============================] - 0s 830us/step - loss: 3.4189
    Epoch 2/40
    16/16 [==============================] - 0s 790us/step - loss: 3.0758
    Epoch 3/40
    16/16 [==============================] - 0s 830us/step - loss: 2.3582
    Epoch 4/40
    16/16 [==============================] - 0s 817us/step - loss: 1.5259
    Epoch 5/40
    16/16 [==============================] - 0s 776us/step - loss: 1.0501
    Epoch 6/40
    16/16 [==============================] - 0s 893us/step - loss: 0.8431
    Epoch 7/40
    16/16 [==============================] - 0s 780us/step - loss: 0.6829
    Epoch 8/40
    16/16 [==============================] - 0s 755us/step - loss: 0.5768
    Epoch 9/40
    16/16 [==============================] - 0s 784us/step - loss: 0.4912
    Epoch 10/40
    16/16 [==============================] - 0s 852us/step - loss: 0.4555
    Epoch 11/40
    16/16 [==============================] - 0s 767us/step - loss: 0.4083
    Epoch 12/40
    16/16 [==============================] - 0s 765us/step - loss: 0.3820
    Epoch 13/40
    16/16 [==============================] - 0s 762us/step - loss: 0.3439
    Epoch 14/40
    16/16 [==============================] - 0s 775us/step - loss: 0.3406
    Epoch 15/40
    16/16 [==============================] - 0s 884us/step - loss: 0.3211
    Epoch 16/40
    16/16 [==============================] - 0s 771us/step - loss: 0.3098
    Epoch 17/40
    16/16 [==============================] - 0s 822us/step - loss: 0.2892
    Epoch 18/40
    16/16 [==============================] - 0s 743us/step - loss: 0.2598
    Epoch 19/40
    16/16 [==============================] - 0s 719us/step - loss: 0.2494
    Epoch 20/40
    16/16 [==============================] - 0s 767us/step - loss: 0.2663
    Epoch 21/40
    16/16 [==============================] - 0s 737us/step - loss: 0.2536
    Epoch 22/40
    16/16 [==============================] - 0s 698us/step - loss: 0.2230
    Epoch 23/40
    16/16 [==============================] - 0s 761us/step - loss: 0.2281
    Epoch 24/40
    16/16 [==============================] - 0s 735us/step - loss: 0.2143
    Epoch 25/40
    16/16 [==============================] - 0s 736us/step - loss: 0.2039
    Epoch 26/40
    16/16 [==============================] - 0s 769us/step - loss: 0.2036
    Epoch 27/40
    16/16 [==============================] - 0s 747us/step - loss: 0.1979
    Epoch 28/40
    16/16 [==============================] - 0s 739us/step - loss: 0.2157
    Epoch 29/40
    16/16 [==============================] - 0s 791us/step - loss: 0.1998
    Epoch 30/40
    16/16 [==============================] - 0s 774us/step - loss: 0.2208
    Epoch 31/40
    16/16 [==============================] - 0s 784us/step - loss: 0.2367
    Epoch 32/40
    16/16 [==============================] - 0s 760us/step - loss: 0.1999
    Epoch 33/40
    16/16 [==============================] - 0s 781us/step - loss: 0.1901
    Epoch 34/40
    16/16 [==============================] - 0s 785us/step - loss: 0.1903
    Epoch 35/40
    16/16 [==============================] - 0s 753us/step - loss: 0.1871
    Epoch 36/40
    16/16 [==============================] - 0s 823us/step - loss: 0.2021
    Epoch 37/40
    16/16 [==============================] - 0s 814us/step - loss: 0.1823
    Epoch 38/40
    16/16 [==============================] - 0s 805us/step - loss: 0.1743
    Epoch 39/40
    16/16 [==============================] - 0s 781us/step - loss: 0.1741
    Epoch 40/40
    16/16 [==============================] - 0s 762us/step - loss: 0.1751
    313/313 [==============================] - 0s 594us/step - loss: 0.2125





    0.2125217467546463




```python
model_cor = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=(3,)),
    keras.layers.Dense(10, activation="relu", input_shape=(3,)),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model_cor.compile(optimizer=optimizer, loss="mse")
model_cor.fit(X_cor_train, Y_cor_train, epochs=40)
model_cor.evaluate(X_cor_test, Y_cor_test)
```

    Epoch 1/40
    16/16 [==============================] - 0s 807us/step - loss: 3.1813
    Epoch 2/40
    16/16 [==============================] - 0s 746us/step - loss: 1.4841
    Epoch 3/40
    16/16 [==============================] - 0s 818us/step - loss: 0.8389
    Epoch 4/40
    16/16 [==============================] - 0s 815us/step - loss: 0.5696
    Epoch 5/40
    16/16 [==============================] - 0s 842us/step - loss: 0.3757
    Epoch 6/40
    16/16 [==============================] - 0s 821us/step - loss: 0.2766
    Epoch 7/40
    16/16 [==============================] - 0s 796us/step - loss: 0.2243
    Epoch 8/40
    16/16 [==============================] - 0s 863us/step - loss: 0.1953
    Epoch 9/40
    16/16 [==============================] - 0s 881us/step - loss: 0.1722
    Epoch 10/40
    16/16 [==============================] - 0s 787us/step - loss: 0.1589
    Epoch 11/40
    16/16 [==============================] - 0s 787us/step - loss: 0.1502
    Epoch 12/40
    16/16 [==============================] - 0s 749us/step - loss: 0.1409
    Epoch 13/40
    16/16 [==============================] - 0s 746us/step - loss: 0.1364
    Epoch 14/40
    16/16 [==============================] - 0s 770us/step - loss: 0.1367
    Epoch 15/40
    16/16 [==============================] - 0s 747us/step - loss: 0.1335
    Epoch 16/40
    16/16 [==============================] - 0s 822us/step - loss: 0.1291
    Epoch 17/40
    16/16 [==============================] - 0s 766us/step - loss: 0.1264
    Epoch 18/40
    16/16 [==============================] - 0s 777us/step - loss: 0.1300
    Epoch 19/40
    16/16 [==============================] - 0s 770us/step - loss: 0.1264
    Epoch 20/40
    16/16 [==============================] - 0s 771us/step - loss: 0.1279
    Epoch 21/40
    16/16 [==============================] - 0s 766us/step - loss: 0.1369
    Epoch 22/40
    16/16 [==============================] - 0s 803us/step - loss: 0.1243
    Epoch 23/40
    16/16 [==============================] - 0s 829us/step - loss: 0.1206
    Epoch 24/40
    16/16 [==============================] - 0s 802us/step - loss: 0.1171
    Epoch 25/40
    16/16 [==============================] - 0s 822us/step - loss: 0.1177
    Epoch 26/40
    16/16 [==============================] - 0s 853us/step - loss: 0.1191
    Epoch 27/40
    16/16 [==============================] - 0s 761us/step - loss: 0.1187
    Epoch 28/40
    16/16 [==============================] - 0s 767us/step - loss: 0.1177
    Epoch 29/40
    16/16 [==============================] - 0s 785us/step - loss: 0.1228
    Epoch 30/40
    16/16 [==============================] - 0s 828us/step - loss: 0.1240
    Epoch 31/40
    16/16 [==============================] - 0s 790us/step - loss: 0.1200
    Epoch 32/40
    16/16 [==============================] - 0s 788us/step - loss: 0.1172
    Epoch 33/40
    16/16 [==============================] - 0s 869us/step - loss: 0.1133
    Epoch 34/40
    16/16 [==============================] - 0s 755us/step - loss: 0.1128
    Epoch 35/40
    16/16 [==============================] - 0s 761us/step - loss: 0.1204
    Epoch 36/40
    16/16 [==============================] - 0s 768us/step - loss: 0.1160
    Epoch 37/40
    16/16 [==============================] - 0s 883us/step - loss: 0.1109
    Epoch 38/40
    16/16 [==============================] - 0s 812us/step - loss: 0.1247
    Epoch 39/40
    16/16 [==============================] - 0s 780us/step - loss: 0.1154
    Epoch 40/40
    16/16 [==============================] - 0s 799us/step - loss: 0.1139
    313/313 [==============================] - 0s 596us/step - loss: 0.1369





    0.13692966103553772



---
## PDP - Influence of feature interactions and feature correlations



Let' s estimate some notation for the rest of the tutorial:

<center>

| Symbol                                                     | Description                                             |
|------------------------------------------------------------|---------------------------------------------------------|
| $f(\mathbf{x})$                                            | The black box model                                     |
| $x_s$                                                      | The feature of interest                                 |
| $x_c$                                                      | The remaining features, i.e., $\mathbf{x} = (x_s, x_c)$ |
| $\mathbf{x} = (x_s, x_c) = (x_1, x_2, ..., x_s, ..., x_D)$ | The input features                                      |
| $\mathbf{x}^{(i)} = (x_s^{(i)}, x_c^{(i)})$                | The $i$-th instance of the dataset                      |

The PDP is defined as **_the average of the model's prediction over the entire dataset, while varying the feature of interest._**
PDP is defined as 

$$ \text{PDP}(x_s) = \mathbb{E}_{x_c}[f(x_s, x_c)] $$ 

and is approximated by 

$$ \hat{\text{PDP}}(x_s) = \frac{1}{N} \sum_{j=1}^N f(x_s, x^{(i)}_c) $$

Therfore, the PDP is an verage over the underlying ICE curves (local effects) which visualize how the feature of interest influences the prediction of the ML model for each single instance. Heterogeneous ICE curves indicate feature interactions. Therefore, we are expecting heterogeneous ICE curves for $x_1$ and $x_3$ for our uncorrelated simulation example, which can be explained by the underlying feature interactions.

Let's check it out the PDP effect using `effector`.

### Uncorrelated setting


```python
pdp = effector.PDP(data=X_uncor_train, model=model_uncor, feature_names=['x1','x2','x3'], target_name="Y")
fig, ax = pdp.plot(feature=0, centering=True, show_avg_output=False, confidence_interval="ice")
fig, ax = pdp.plot(feature=1, centering=True, show_avg_output=False, confidence_interval="ice")
fig, ax = pdp.plot(feature=2, centering=True, show_avg_output=False, confidence_interval="ice")
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_11_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_11_1.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_11_2.png)
    


In the uncorrelated setting $x_1$ has - as expected - an average global feature effect of $0$ while the heteroeneous ICE curves indicate the underlying feature interactions with $x_3$. Also $x_3$ demonstrates a high heterogeneity of local effects while the average global effect represents the underlying main effect of $x_3$. $x_2$ varies as expected around $0$ with small heterogeneity (note the difference in the scale of the y-axis).

#### GADGET-PD

GADGET creates interpretable and distinct regions within the feature space such that the interaction-related heterogeneity is minimized within the regions. Thus, we aim to receive main effects of the features within the regions. 


In the case of PDPs and ICE this means, that we minimize the heterogeneity of mean-centered ICE curves. This means that we group ICE curves with a similar shape, i.e., we find regions in which the instances within this regions show a similar influence on the prediction for the feature of interests, while this influence differs for other regions.


```python
regional_pdp = effector.RegionalPDP(data=X_uncor_train, model=model_uncor, feature_names=['x1','x2','x3'], axis_limits=np.array([[-1,1],[-1,1],[-1,1]]).T)

regional_pdp.fit(
    features="all",
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.3,
    max_split_levels=2,
    nof_candidate_splits_for_numerical=5,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)
```

    100%|██████████| 3/3 [00:00<00:00,  3.64it/s]



```python
regional_pdp.describe_subregions(features=0, only_important=True)
```

    Important splits for feature x1
    - On feature x3 (cont)
      - Range: [-1.00, 1.00]
      - Candidate split positions: -0.80, -0.40, -0.00, 0.40, 0.80
      - Position of split: -0.00
      - Heterogeneity before split: 1.76
      - Heterogeneity after split: 0.39
      - Heterogeneity drop: 1.38 (355.35 %)
      - Number of instances before split: 500
      - Number of instances after split: [255, 245]



```python
regional_pdp.plot_first_level(feature=0, confidence_interval="ice")
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_17_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_17_1.png)
    



```python
regional_pdp.describe_subregions(features=1, only_important=True)
```

    No important splits found for feature 1



```python
regional_pdp.describe_subregions(features=2, only_important=True)
```

    Important splits for feature x3
    - On feature x1 (cont)
      - Range: [-1.00, 1.00]
      - Candidate split positions: -0.80, -0.40, -0.00, 0.40, 0.80
      - Position of split: -0.00
      - Heterogeneity before split: 1.73
      - Heterogeneity after split: 0.87
      - Heterogeneity drop: 0.86 (99.31 %)
      - Number of instances before split: 500
      - Number of instances after split: [257, 243]



```python
regional_pdp.plot_first_level(feature=2, confidence_interval="ice", centering=True)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_20_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_20_1.png)
    


### Correlated setting

GADGET creates interpretable and distinct regions within the feature space such that the interaction-related heterogeneity is minimized within the regions. Thus, we aim to receive main effects of the features within the regions. 
Since the PDP assumes feature independence, we can observe in the highly correlated setting the following artifact: $x_1$ and $x_3$ are highly positively correlated, therefore, the combination of small (high) $x_1$ and high (small) $x_3$ feature values is not avaiable and thus has not been seen by the model during the training process. However, ICE curves and PDPs are visualized for the entire feature range of the feature of interest (e.g., $x_1$). Thus, we extrapolate with our model (here NN) into unseen or sparse regions of the feature space. This might lead to an osciliating behavior depending on the underlying chosen ML model. Therefore, we might receive heterogeneity of local effects (ICE curves) which are not caused by feature interactions but by extrapolation due to feature correlations. This behavior is especially visible for feature $x_1$ in our example.   





```python
pdp = effector.PDP(data=X_cor_train, model=model_cor, feature_names=['x1','x2','x3'], target_name="Y")
fig, ax = pdp.plot(feature=0, centering=True, show_avg_output=False, confidence_interval="ice")
fig, ax = pdp.plot(feature=1, centering=True, show_avg_output=False, confidence_interval="ice")
fig, ax = pdp.plot(feature=2, centering=True, show_avg_output=False, confidence_interval="ice")
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_22_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_22_1.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_22_2.png)
    


#### GADEGT-PD


```python
regional_pdp = effector.RegionalPDP(
    data=X_cor_train, 
    model=model_cor, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)

regional_pdp.fit(
    features="all",
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.3,
    max_split_levels=2,
    nof_candidate_splits_for_numerical=5,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)
```

    100%|██████████| 3/3 [00:00<00:00,  3.69it/s]



```python
regional_pdp.describe_subregions(features=0, only_important=True)
```

    Important splits for feature x1
    - On feature x3 (cont)
      - Range: [-1.00, 1.00]
      - Candidate split positions: -0.80, -0.40, -0.00, 0.40, 0.80
      - Position of split: -0.00
      - Heterogeneity before split: 1.52
      - Heterogeneity after split: 0.41
      - Heterogeneity drop: 1.10 (266.53 %)
      - Number of instances before split: 500
      - Number of instances after split: [255, 245]



```python
regional_pdp.plot_first_level(feature=0, confidence_interval="ice")
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_26_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_26_1.png)
    



```python
regional_pdp.describe_subregions(features=1, only_important=True)
```

    Important splits for feature x2
    - On feature x1 (cont)
      - Range: [-1.79, 1.37]
      - Candidate split positions: -1.47, -0.84, -0.21, 0.42, 1.05
      - Position of split: 0.42
      - Heterogeneity before split: 1.39
      - Heterogeneity after split: 0.85
      - Heterogeneity drop: 0.53 (62.51 %)
      - Number of instances before split: 500
      - Number of instances after split: [360, 140]



```python
regional_pdp.plot_first_level(feature=1, confidence_interval="ice", centering=True)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_28_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_28_1.png)
    



```python

```


```python
regional_pdp.describe_subregions(features=2, only_important=True)
```

    Important splits for feature x3
    - On feature x1 (cont)
      - Range: [-1.79, 1.37]
      - Candidate split positions: -1.47, -0.84, -0.21, 0.42, 1.05
      - Position of split: -0.21
      - Heterogeneity before split: 1.27
      - Heterogeneity after split: 0.79
      - Heterogeneity drop: 0.48 (61.30 %)
      - Number of instances before split: 500
      - Number of instances after split: [192, 308]



```python
regional_pdp.plot_first_level(feature=2, confidence_interval="ice", centering=True)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_31_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_31_1.png)
    


## ALE - Influence of feature interactions and feature correlations





```python
def model_uncor_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model_uncor(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()

def model_cor_jac(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        pred = model_cor(x_tensor)
        grads = t.gradient(pred, x_tensor)
    return grads.numpy()
```


```python
rhale = effector.RHALE(data=X_uncor_train, model=model_uncor, model_jac=model_uncor_jac, feature_names=['x1','x2','x3'], target_name="Y")

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=10)
rhale.fit(features="all", binning_method=binning_method, centering=True)

rhale.plot(feature=0, centering=True, show_avg_output=False)
rhale.plot(feature=1, centering=True, show_avg_output=False)
rhale.plot(feature=2, centering=True, show_avg_output=False)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_34_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_34_1.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_34_2.png)
    



```python
regional_rhale = effector.RegionalRHALE(
    data=X_uncor_train, 
    model=model_uncor, 
    model_jac= model_uncor_jac, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T) 

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=10)
regional_rhale.fit(
    features="all",
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.3,
    max_split_levels=2,
    nof_candidate_splits_for_numerical=5,
    binning_method=binning_method,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)

```

    100%|██████████| 3/3 [00:00<00:00,  9.74it/s]



```python
regional_rhale.describe_subregions(features=0, only_important=True)
```

    Important splits for feature x1
    - On feature x3 (cont)
      - Range: [-1.00, 1.00]
      - Candidate split positions: -0.80, -0.40, -0.00, 0.40, 0.80
      - Position of split: -0.00
      - Heterogeneity before split: 5.87
      - Heterogeneity after split: 1.36
      - Heterogeneity drop: 4.51 (332.64 %)
      - Number of instances before split: 500
      - Number of instances after split: [255, 245]



```python
regional_rhale.plot_first_level(
    feature=0, 
    confidence_interval=True, 
    binning_method=binning_method)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_37_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_37_1.png)
    



```python
regional_rhale.describe_subregions(features=1, only_important=True)
```

    No important splits found for feature 1



```python
regional_rhale.describe_subregions(features=2, only_important=True)
```

    No important splits found for feature 2


## Correlated setting


```python
rhale = effector.RHALE(data=X_cor_train, model=model_cor, model_jac=model_cor_jac, feature_names=['x1','x2','x3'], target_name="Y")

binning_method = effector.binning_methods.Fixed(10, min_points_per_bin=0)
rhale.fit(features="all", binning_method=binning_method, centering=True)
```


```python
rhale.plot(feature=0, centering=True, show_avg_output=False)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_42_0.png)
    



```python
rhale.plot(feature=1, centering=True, show_avg_output=False)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_43_0.png)
    



```python
rhale.plot(feature=2, centering=True, show_avg_output=False)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_44_0.png)
    



```python
regional_rhale = effector.RegionalRHALE(
    data=X_cor_train, 
    model=model_cor, 
    model_jac= model_cor_jac, 
    feature_names=['x1', 'x2', 'x3'],
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]).T)
```


```python
binning_method = effector.binning_methods.Greedy(100, min_points_per_bin=10)
regional_rhale.fit(
    features="all",
    heter_small_enough=0.1,
    heter_pcg_drop_thres=0.3,
    max_split_levels=2,
    nof_candidate_splits_for_numerical=5,
    binning_method=binning_method,
    min_points_per_subregion=10,
    candidate_conditioning_features="all",
    split_categorical_features=True,
)
```

    100%|██████████| 3/3 [00:01<00:00,  2.64it/s]



```python
regional_rhale.describe_subregions(features=0, only_important=True)
```

    Important splits for feature x1
    - On feature x3 (cont)
      - Range: [-1.00, 1.00]
      - Candidate split positions: -0.80, -0.40, -0.00, 0.40, 0.80
      - Position of split: -0.00
      - Heterogeneity before split: 1.88
      - Heterogeneity after split: 1.04
      - Heterogeneity drop: 0.84 (80.92 %)
      - Number of instances before split: 500
      - Number of instances after split: [255, 245]



```python
regional_rhale.plot_first_level(
    feature=0, 
    confidence_interval=True, 
    binning_method=binning_method)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_48_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_48_1.png)
    



```python
regional_rhale.describe_subregions(features=1, only_important=True)
```

    No important splits found for feature 1



```python
regional_rhale.describe_subregions(features=2, only_important=True)
```

    Important splits for feature x3
    - On feature x1 (cont)
      - Range: [-1.79, 1.37]
      - Candidate split positions: -1.47, -0.84, -0.21, 0.42, 1.05
      - Position of split: -0.21
      - Heterogeneity before split: 1.66
      - Heterogeneity after split: 1.11
      - Heterogeneity drop: 0.56 (50.60 %)
      - Number of instances before split: 500
      - Number of instances after split: [192, 308]



```python
regional_rhale.plot_first_level(
    feature=2, 
    confidence_interval=True, 
    binning_method=binning_method)
```


    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_51_0.png)
    



    
![png](04_regional_effects_pdp_vs_rhale_files/04_regional_effects_pdp_vs_rhale_51_1.png)
    



```python

```
