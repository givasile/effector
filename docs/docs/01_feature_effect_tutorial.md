---
title: Feature Effect Methods
---

???+ question "Why we care about feature effect plots?"
     
     Because they are (probably) the simplest way to globally interpret a black-box model.


Imagine you have trained a neural network to predict the expected daily bike rentals,
like we do in this [tutorial](./tutorials/03_bike_sharing_dataset.md).
The model is delivering satisfactory results, 
exhibiting an average prediction error of approximately $81$ bike rentals per day.

You want to interpret how the model works. 
With feature effect plots, you can immediately get a graphical representation that illustrates 
how individual features impact the model's predictions.

![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_13_0.png)

The plot shows the effect of feature $x_4$ which is the $hour$ of the day on the prediction.
With this plots, we can get an immediate *interpretation* of the model's inner workings,
which can raise some *criticism* and lead to appropriate *actions*.

???+ note "Interpretation: Move along the axis and interpret"
     
    | Interval  | Description                                                              |
    |-----------|--------------------------------------------------------------------------|
    | 0-6       | The hour is almost constant and much lower than the average.             |
    | 6-8.30    | The hour is increasing rapidly and at about 7.30 it is over the average. |
    | 8.30-9.30 | Drop that moves the rentals close to the average.                        |
    | 9.30-15   | Small increase.                                                          |
    | 15-17     | High increase, moves much over the average.                              |
    | 17-24     | A constant drop; faster in the beginning slower later.                   |

--- 

???+ note "Criticism 1: Does this makes sense?"

    It seems like a normal working day. Most people go to work between 6' and 8'30" o'clock and 
    come back between 15'00" and 17'00 o'clock. 
    An expert in city transfers, could raise the concern that the model is slightly shifted early;   
    He/She may say that the peak of the rentals is at 8'30" and not at 7'30" o'clock.

???+ note "Action 1: Investigate more"    
     We could investigate whether this a problem of the model or of the data; Does the data confirm 
     the above statement? If yes, we should improve the model to increase its accuracy. 

---

???+ note "Criticism 2: Is the explanation correct?"

    Another expert may notice that this behavior is meaningful only for the working days.
    At weekends and holidays, it does not make sense to have a peak at 8'30" o'clock.

???+ note "Action 2: Who to blame? The models or the explanation?"
    To take action, we should clarify whether this is a problem of the model or of the explanation.
    If it is a problem of the model, we should improve it to increase its accuracy on the weekends.
    But how to know if it is a problem of the model or of the explanation?

### Heterogeneity shows the fidelity of the explanation

Let's see whether the explanation is consistent with all the data:


```python
effector.RHALE(X, model, model_jac).plot(feature=3, confidence_interval=True)
```

![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_16_0.png)

```python
effector.PDPwithICE(X, model).plot(feature=3)
```

![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_17_0.png)


Both methods show that there is high-variance in the instance-level effect.
This means that although the global explanation is the one analyzed above,
there are many individual instances that deviate from this explanation.

In fact, PDP-ICE shows the exact type of the different behaviors:

- One cluster, the dominant one, which is the working days, behave as we have seen before and in a more edgy way.
- A second cluster, the weekends, behave differently, with a peak at 12' o'clock and a drop at 18' o'clock.

???+ attention "We have an answer"
    It is not a problem of the model, but that the global explanation hided the two patterns behind the averaging.

[Regional effect plots](./02_regional_effect_tutorial.md) automate the process of finding such patterns.

---
### Resources for further reading

Below we provide some resources for further reading:

Papers:

- [Model-Agnostic Effects Plots for Interpreting Machine Learning Models](http://www1.beuth-hochschule.de/FB_II/reports/Report-2020-001.pdf) 


Books:

- [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
- [https://ema.drwhy.ai/preface.html](https://ema.drwhy.ai/preface.html)
- [Limitations of Interpretable Machine Learning Methods](https://slds-lmu.github.io/iml_methods_limitations/)