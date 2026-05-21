# Assignment7: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay

# Task 1 - Caoimhe
We fit a prophet model to the data this time, to better account for weekly and yearly seasonality, changing trends with time, and holidays. First we fit it to the daily time series, after removing weekends by using the business day frequency, and cutting the data to start 30 days in, once lots of journeys were up and running and the average number of orders per day was above 0 and 1. Since prophet is additive, it often predicts negative values for y, which is obviously nonsensical in this case. To avoid this issue, we converted to a log scale while fitting the model, and then converted back to the real values for plotting and interpretation. The results for this model's prediction can be seen here (2023_prophet.png). The breakdown of the individual components can be seen here (2023_prophet_components.png), with a consistent downward trend, both weekly and seasonal components, and a holiday component. We can further investigate the model by looking at some of the parameters (prophet_coef.png). We can see that there is only one change point in the trend, which is also evident in the component plots, where we can see that the trend changes slightly in August 2022, but remains a negative slope always. The volatility of the changes in the trend is very small, meaning that the trend component is smooth. 

We chose to use a daily model rather than monthly, as there is clear weekly seasonality in the data. The predictions for daily orders shipped can be found in the the plots listed above, and are also provided in this csv along with the upper and lower bounds for the confidence interval (time_series_predictions.csv)

# Task 2 - 

# Task 3 - Derek

Using my XGBoost model, I was able to model the success probability of 2 specific journeys, one resulting in a success and one resulting in a failure. To accurately do this, I had to find the features which were time dependent (e.g. time since last event) and adjust them to match the inactivity or activity seen throughout the journey, to then create predictions. Otherwise, there would have been 0 change in predicted probability between events due to lack of inactivity since the features reflecting that would be static.


![The joint probability for a successful journey](joint_probability_timeline_successful.png)
![The joint probability for an unsuccessful journey](joint_probability_timeline_unsuccessful.png)


# Task 4 - 