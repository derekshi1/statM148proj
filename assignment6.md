# Assignment 6: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay

# Task 1
We fit a prophet model to the data this time, to better account for weekly and yearly seasonality, changing trends with time, and holidays. First we fit it to the daily time series, after removing weekends by using the business day frequency, and cutting the data to start 30 days in, once lots of journeys were up and running and the average number of orders per day was above 0 and 1. Since prophet is additive, it often predicts negative values for y, which is obviously nonsensical in this case. To avoid this issue, we converted to a log scale while fitting the model, and then converted back to the real values for plotting and interpretation. The results for this model's prediction can be seen here (2023_prophet.png). The breakdown of the individual components can be seen here (2023_prophet_components.png), with a consistent downward trend, both weekly and seasonal components, and a holiday component. We can further investigate the model by looking at some of the parameters (prophet_coef.png). We can see that there is only one change point in the trend, which is also evident in the component plots, where we can see that the trend changes slightly in August 2022, but remains a negative slope always. The volatility of the changes in the trend is very small, meaning that the trend component is smooth. 

We can create the same model for the monthly time series, and predict a year into the future again (monthly_prophet.png). Interestingly the trend in this model is increasing (monthly_components). The parameters for this model are similar to the daily model, with only one change point in the trend and very small volatility indicating a smooth trend again (monthly_params.png). 

Overall, the daily model is likely more accurate, as we can include weekly seasonality as well as holidays.

# Task 2
Our best model in the kaggle competition was an ensemble of models that Jayden made, using XGBoost and RandomForest. He was able to effectively tune hyperparameters using an engineered validation set which closer matches the testing set. Other models we have tried include a transformer, temporal point processing, and continuous time markov chains.

Looking forward, we hope to train
the transformer on a much smaller subset of data, to avoid high training time and cost. 

Another thing we want to implement is more feature engineering, and better feature selection for the final model.

Additionally, we hope to have more effective hyperparameter tuning, by being very careful with our validation set, finally try using an ensemble of different models than what we used originally.

# Task 3

We took a break from implmeneting transforemers and temporal point processing, so I tried implementing continuous time markov chains-- based on Cash's success of using thme. The three states I used are active successful and unsuccessful, where the absorving states are success and unsuccess. To create the transition matrix, I used 9 features that i selected from correlation analysis: mean gap, total elapsed time, event count, unique event count, max milestone, downpayment count, prospecting count, last event id, and events in the last 3 days. I used HistGradientBoosting Classifier as the feature model with parameters of lr, max nodes, regularization, and min samples per leaf.

After validating the model across held out data, it is usually right when a journey is unsuccessful, but it misses a lot of the true successes.

