# Assignment 6: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay

# Task 1

# Task 2
Our best model in the kaggle competition was an ensemble of models that Jayden made, using XGBoost and RandomForest. He was able to effectively tune hyperparameters using an engineered validation set which closer matches the testing set. Other models we have tried include a transformer, temporal point processing, and continuous time markov chains.

Looking forward, we hope to train
the transformer on a much smaller subset of data, to avoid high training time and cost. 

Another thing we want to implement is more feature engineering, and better feature selection for the final model.

Additionally, we hope to have more effective hyperparameter tuning, by being very careful with our validation set, finally try using an ensemble of different models than what we used originally.

# Task 3

We took a break from implmeneting transforemers and temporal point processing, so I tried implementing continuous time markov chains-- based on Cash's success of using thme. The three states I used are active successful and unsuccessful, where the absorving states are success and unsuccess. To create the transition matrix, I used 9 features that i selected from correlation analysis: mean gap, total elapsed time, event count, unique event count, max milestone, downpayment count, prospecting count, last event id, and events in the last 3 days. I used HistGradientBoosting Classifier as the feature model with parameters of lr, max nodes, regularization, and min samples per leaf.

After validating the model across held out data, it is usually right when a journey is unsuccessful, but it misses a lot of the true successes.

