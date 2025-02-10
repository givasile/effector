---
title: Global and Regional Effects
---

## Global effects

???+ question "Why do we care about global effects?"
     
     Because they are super simple.

✅ You trained a neural network (1) to predict hourly bike rentals using historical data. The dataset includes features like `hour`, `weekday`, `workingday`, `temperature`, `humidity`, and, of course, the `bike-rentals`.
{ .annotate }  

1. 📌 You can find the full notebook [here](./../../notebooks/real-examples/01_bike_sharing_dataset/).

🚀 The model performs well, with an average prediction error of about 43 bikes. Now, you want to understand how it makes predictions.

📊 Feature effect plots provide a visual way to see how each feature influences the model's output

=== "`month`"
    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_0.png)

=== "`hour`"
    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_1.png)
    
=== "`temperature`"
    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_2.png)
    
=== "`humidity`"
    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_3.png)
    
=== "`windspeed`"
    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_4.png)

Interesting! The model has a meaningful behavior:

- 🔍 month, humidity, and windspeed have little impact. 
- 🌡️ Temperature has a stronger positive effect, but 
- ⏰ hour is the most important feature; let's focus on that

---

Let's focus on feature `hour` and analyze with more global effect methods.

=== "PDP"
    ```python
    effector.PDP(X, model).plot(feature=3)
    ```

    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_18_1.png)

=== "RHALE"
    ```python
    effector.RHALE(X, model, model_jac).plot(feature=3)
    ```

    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_22_0.png)

=== "SHAP-DP"

    ```python
    effector.ShapDP(X, model).plot(feature=3)
    ```

    ![Feature effect plot](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_23_1.png)


All methods aggree on the effect;
there is an abrupt increase in the number of bike rentals at about 8:00 AM (beginning of the workday)
and at about 5:00 PM (end of the workday). 
The following table provides a more detailed interpretation of the plot: 

???+ note "Interpretation: Move along the axis and interpret"
     
    | Interval  | Description                                                                              |
    |-----------|------------------------------------------------------------------------------------------|
    | 0-6       | Bike rentals are almost constant and much lower than the average, which is $\approx 189$ |
    | 6-8.30    | Rapid increase; at about 7.00 we are at the average rentals and then even more.          |
    | 8.30-9.30 | Sudden drop; rentals move back to the average.                                           |
    | 9.30-15   | Small increase.                                                                          |
    | 15-17     | High increase; at 17 bike rentals reach the maximum.                                     |
    | 17-24     | A constant drop; at 19.00 rentals reach the average and keep decreasing.                 |

--- 

Global feature effect plots provide an immediate *interpretation* of the model's inner workings.


???+ question "Criticism 1: Does this makes sense?"

     It seems reasonable. On a typical workday, people commute between 6:00–8:30 AM and return between 3:00–5:00 PM. But a city transportation expert might have a better perspective.

???+ question "Criticism 2: Is the explanation valid in all cases?"

    An expert might point out that this pattern makes sense only on working days. On weekends and holidays, an early peak at about 7:30 AM wouldn't be as logical.

---

## Heterogeneity

Criticism 2 questions whether the explanation applies to the entire dataset. Let's check the heterogeneity:

🔴 **red ICE curves in PDP**  
🔴 **red bars in RHALE plots**  
🔴 **red SHAP values in SHAP-DP plots**  

These indicate that the global effect might be masking cases that deviate from the overall trend.
Moreover, PDP-ICE analysis highlights two distinct patterns:

- There is one cluster, that behaves as described above. 
- There is a second cluster that behaves differently, with a rise starting at 9:00 AM, a peak at 12:00 AM and a decline at 6:00 PM.

???+ danger "Don't rush to conclusions"

    There is a small piece of the puzzle missing.
    Although we have identified the two distinct patterns, we still don't know what causes them.
    Of course, we can guess that the first pattern is related to the working days, and the second pattern is related to the weekends and holidays.
    But this is simply our intuition, and we need to confirm it with the data. 
    We need to find the features that are responsible for the two distinct patterns.
    [Regional effect plots](./02_regional_effect_intro.md) are the answer to this question.

---

## Regional effects

???+ question "Why do we care about regional effects?"

    Because they are super simple and provide richer information that global effects.
    
???+ Note "When Global Effect is a weak explanation?"

    In cases where the global effect plot shows high heterogeneity, it is useful to analyze the regional effect.
    Why is this the case? Because when many instances behave differently from the average pattern, 
    it means that **the effect of feature $x_s$ on the output $y$, depends on the values of other features $x_{\setminus s}$.**
    In these cases, the average effect of feature $x_s$ on the output $y$ is a weak explanation.

???+ Note "When Regional Effect can provide a good solution"

    In cases where the global effect plot shows high heterogeneity, 
    it **may** be the case that there are subregions where the instances behave similarly.
    Regional Effect Plots search for subregions where 
    **the effect of feature $x_s$ on the output $y$, has smaller dependence on the values of other features $x_{\setminus s}$.**

So let's apply regional effect analysis to the $\mathtt{hour}$ feature.
`Effector` provides a simple API for that, similar to the global effect API:

=== "PDP"

     | non-working day and cold | non-workingday and hot |
     |:---------:|:---------:|
     | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_0.png) | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_1.png) |
     | working day and cold | workingday and hot |
     | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_2.png) | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_29_3.png) |


=== "RHALE"

     | non-working day | workingday |
     |:---------:|:---------:|
     | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_32_0.png) | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_32_1.png) |

=== "SHAP-DP"

     | non-working day and cold | non-workingday and hot |
     |:---------:|:---------:|
     | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_0.png) | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_1.png) |
     | working day and cold | workingday and hot |
     | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_2.png) | ![Alt text](../static/real-examples/01_bike_sharing_dataset_files/01_bike_sharing_dataset_37_3.png) |

???+ success "Let's reach some conclusions""

    The regional effect analysis confirms our intuition: the data shows two distinct patterns.  

    📅 **Workdays:** Rentals rise at **8:30 AM** and **5:00 PM**, matching commute times.  
    🌴 **Weekends & Holidays:** Rentals increase at **9:00 AM**, peak at **12:00 PM**, and decline around **4:00 PM**—a typical leisure pattern.  

    📊 **PDP and SHAP-DP go further**  
    They reveal another key factor: **temperature**. The impact of `hour` on bike rentals differs on non-working days depending on whether it’s hot or cold.  

    ✔️ This makes sense—temperature matters for sightseeing, but not for commuting.


---
## Resources for further reading

Below we provide some resources for further reading.

Papers:

- [Model-Agnostic Effects Plots for Interpreting Machine Learning Models](http://www1.beuth-hochschule.de/FB_II/reports/Report-2020-001.pdf) 


Books:

- [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
- [https://ema.drwhy.ai/preface.html](https://ema.drwhy.ai/preface.html)
- [Limitations of Interpretable Machine Learning Methods](https://slds-lmu.github.io/iml_methods_limitations/)
