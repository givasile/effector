# Regional Feature Effect

In the previous tutorial, we explained the effect of feature $\mathtt{hour}$ on the daily $\mathtt{bike-rentals}$, 
using global feature effect plots. 
The analysis, however, showed that there is high heterogeneity; there are many instances that behave differently from the average pattern:

![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_20_0.png)
![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_17_0.png)

---
Regional Feature Effect Plot are the interpretability tool that helps us to understand the heterogeneity in the data;
they try to automatically identify the groups of instances that behave similarly and 
show the effect of the feature on each group separately. 
For example, if we apply regional effect analysis to the $\mathtt{hour}$ feature, we will get the following 
explanation:

```python

```

![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_25_1.png)
![Feature effect plot](./tutorials/03_bike_sharing_dataset_files/03_bike_sharing_dataset_25_2.png)

The above plots show that there are two distinct patterns in the data:
one pattern is related to the working days, and the second pattern is related to the weekends and holidays.
The first pattern is characterized by a rise in the number of rentals at 8:30 AM andn 
at 17:00 AM, when people go to and from work.
The second pattern is characterized by a rise in the number of rentals at 9:00 AM, a peak at 12:00 AM and a decline at 
4:00 PM, a typical non-working day pattern.