#!/usr/bin/env python
# coding: utf-8

# # Model with conditional interaction
# 
# In this example, we show global effects of a model with conditional interactions using PDP, ALE, and RHALE.
# In particular, we:
# 
# 1. show how to use `effector` to estimate the global effects using PDP, ALE, and RHALE
# 2. provide the analytical formulas for the global effects
# 3. test that (1) and (2) match

# We will use the following model: 
# 
# $$ 
# f(x_1, x_2, x_3) = -x_1^2 \mathbb{1}_{x_2 <0} + x_1^2 \mathbb{1}_{x_2 \geq 0} + e^{x_3} 
# $$
# 
# where the features $x_1, x_2, x_3$ are independent and uniformly distributed in the interval $[-1, 1]$.
# 
# 
# The model has an _interaction_ between $x_1$ and $x_2$ caused by the terms: 
# $f_{1,2}(x_1, x_2) = -x_1^2 \mathbb{1}_{x_2 <0} + x_1^2 \mathbb{1}_{x_2 \geq 0}$.
# This means that the effect of $x_1$ on the output $y$ depends on the value of $x_2$ and vice versa.
# Therefore, there is no golden standard on how to split the effect of $f_{1,2}$ to two parts, one that corresponds to $x_1$ and one to $x_2$.
# Each global effect method has a different strategy to handle this issue.
# Below we will see how PDP, ALE, and RHALE handle this interaction.
# 
# In contrast, $x_3$ does not interact with any other feature, so its effect can be easily computed as $e^{x_3}$.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import effector

np.random.seed(21)

model = effector.models.ConditionalInteraction()
dataset = effector.datasets.IndependentUniform(dim=3, low=-1, high=1)
x = dataset.generate_data(1_000)


pdp = effector.PDP(x, model.predict, dataset.axis_limits)
pdp.fit(features="all", centering=True)
for feature in [0, 1, 2]:
    y, h = pdp.eval(feature=0, xs=np.linspace(-1, 1, 100), centering=True, heterogeneity=True)
    print(f"Feature {feature}: {y.std():.2f}, {h.mean():.2f}")
