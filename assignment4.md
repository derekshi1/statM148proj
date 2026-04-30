# Assignment 4: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay



# Task 1 

I chose a 4 journeys for my XGB model to create CP profiles. They represent borderline cases (0.5 predicted success, and eventual success), high risk of a false positive (0.99 predicted success but eventual failure), high chance of failure (eventual failure), and high chance of success (eventual success)

First plot, showing the borderline case of a 0.5 predicted and eventual success
![](borderline_incomplete_cp.png)

Second plot, showing the high risk of a false positive (0.995 probability of success, but eventual failure)
![](high_risk_false_positive_cp.png)


Third plot, highly predicted failure at 0.00001 probability of success
![](incomplete_failure_like_cp.png)

Fourth plot, highly predicted success at 0.998 probability of success
![](incomplete_success_like_cp.png)

From these four plots, days since last event, and observed duration of days seems to be the important drivers of success and failure. Both of these features seem to be correlated, where it doesn't make since for the days since last event to be low while the observed duration of days is high. So only when they are both low and both high do we have an interpretable and highly confident prediction of success or otherwise. 

# Task 2 - 
Got a new accuracy of 0.04685 by using an ensemble model with XGBoost, Logistic Regression, LSTM, and a Neural Network.  The Neural Network and the XGBoost probabilities had a correlation of about 99%, so we can continue to explore other model architectures like transformers, CNN, and GRU.

# Task 3 - Caoimhe
The data was loaded again but this time only events that were order shipped were kept, and all columns other than the timestamp were dropped. This gave us a new data frame that could be used for time series analysis. The number of total orders per month can be seen in this bar chart (monthly_orders.png), and we can see that there is a general increase at the beginning after only 1 or 2 orders for the first 2 months, and then a spike in December of 2021 reaching over 20,000 orders a month. Then there is a gradual decline in monthly orders throughout 2022, but typically remaining at over 7,500 orders per month.

The same general trends can be seen in the time series plot of the data (time_series.png). There appears to be strong trend and seasonal components in the data, so before it can be modelled we must estimate and remove those.

We first estimate trend by smoothing it for seasonality (trend_component.png), and we are left with the approxximately linear trend within the data, increasing at first, then sloping downwards, with a slight increase again at the end. This trend is then removed from the data before the seasonality is estimated. We propose a weekly and monthly trend in the data and estimate them by averaging the number of orders shipped for each week and month of the year. Once we plot the seasonal components, weekly (weekly_seasonal.png) and monthly (monthly_seasonal.png), we can see a very clear pattern wich indicates that the instict to remove them was correct.

Once the seasonal effects and trend have been subtracted from the data, we see the true randomness and can evaluate it to fit a model. In this case, it appears to be simply wite noise with no autoregressive or moving average components, however there is clear heteroskedasticity (random_component.png). To account for this, we fit a garch model to forecast the variances for the future year. These are used to generate future values for the random component of the data, these values are normally distributes with mean zero and the variances calculated previously. 

Once this is done, we can combine all 3 components to forecast the number of orders shipped per day for the remainder of 2023. First, we had split our data into testing and training data to validate the forecast before expanding it to the future. The results of that validation forecast can be seen here (validation_forecast.png). The forecast for 2023 can be found here (2023_forecast.png), and it appears to fit the general declining trend, as well as the changing variation of the original data. It appears to fail around December, where there have been peaks in previous years but not in this forecast. We hope to improve this in future models, but it may be difficult as we only have 15 months of data, so we cannot be sure if this is a genuine annual trend. For example, December 2021 sees a huge spike in orders, but we don't know if that it due to Christmas or due to the site taking off 3 months after launch. 

# Task 4 - Derek

I tried using Temporal Point Process (TPP) because it is designed to predict the timing and nature of discrete events within a continuous time window, differing from RNN and LSTM. TPP treats the journey as a dynamic intensity, modeling the probability of a success event occurring given the user's history and the time elapsed since their last action.To make this work with our specific dataset, we transformed raw customer milestones into sequences of event IDs and log-scaled time deltas. We shifted the targets: at any point $i$ in a journey, the model is trained to predict the event type and the specific time gap for step $i+1$. We designed it this way with the "real-time" requirement in mind in the testing set. Because we model the decay of intent over time, the model can naturally distinguish between a "hot" lead who is actively clicking and a "cold" lead who has been idle for days. 

A primary consideration was the 95/5 class imbalance in testing, which we addressed using weighted cross-entropy and sampling to ensure the model prioritizes the rare "Order Shipped" signal. We also accounted for the fact that journeys in the test set are unfinished by utilizing the TPP's survival function, which allows us to calculate the probability of a future shipment without requiring the journey to be artificially truncated.

There were also many considerations in terms of where we would pad our sequences (start end?), and the max_len we would pad our sequences to. We are working on having this model train successfully, and will test it in the the next weeks hopefully!
