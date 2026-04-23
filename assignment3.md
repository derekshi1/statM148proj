# Task 1
I am not sure how useful this plot is, as the amount of unsuccessful users makes it much harder to create a distinction in the feature's effect on success. We are working on subsetting the pdp plot to show the success vs. non success users and isolate the feature affect that way.

![Ice plot analyzing the effect of "max_milesone_seen", which was the most important predictor in my xgboost model](pdp_ice_plot.png)

# Task 2

In the dataset, "Incomplete" (Unsuccessful) journeys are defined by a lack of action for 60 days, whereas "Successful" journeys end with an order_shipped event. However, a "completed" successful journey is often much shorter in duration than the 60-day threshold required to declare a failure.

To create a fair comparison and avoid a look-ahead Bias (predicting the past using future information), we implemented two specific strategies: **Proportional Weighting and Random Truncation**.

Successful journeys in our data have a median duration of 18 days, while unsuccessful journeys linger for a median of 159 days.

In a real-world environment, we are much more likely to "catch" a long journey in progress than a short one. Logically, a user in a 100 day journey is 100 times more likely to be int he testing set than a use rin a single day journey.

To ensure our training data reflects this "Testing Distribution," we did not sample users uniformly. Instead we create a feature called sampling_weight, which will create "importance" for longer journeys as a parameter our machine learning model. 


Then we apply the Random Truncation, to simulate the random ability to be caught in the middle of a journey, rather than including the successful actions itself.

Finally, we add the momentum feature which is encoded across 1d, 3d, and 5days. This showed some sort of signal between success and non successful users in EDA, so we hope it is a high impact feature. 

# Task 3
First we fit a model with all features included before determining which are of great or little importance to predictions. We are using an XGBoost Model due to the size of the dataset, the presence of numeric and categorical features, as well as sparse features that contain many zeros (such as those that track whether or not a particular event occurs in a particular journey).

XGBoost has lots of possible hyperparameters that control the model's behaviour. The values are selected based on what we know about the data. The model is fitted to the training data with the testing data as a benchmark. We then predict the probability of success for the testing data and compare it to the true outcomes. In this case, any probability above 0.9 is labelled as a successful journey, and below 0.9 is unsuccessful.

This model predicted the correct outcome 85% of the time, so we investigate the individual features to improve this score. The initial feature importance plot (feature_importance.png) is difficult to read as there are 119 features present in this model. To remove unnessecary noise we remove any features with less than 0.005 importance.

This plot (most_important_features.png) is much clearer and shows us that last_stage_Downpayment is by far the most important feature in this model, followed by last_stage_First Purchase, and count_ed_29. The majority of the other features counting if and how often a particular feature appeared are ranked ery low, so it is interesting that evend 29 is so important. This code corresponds to account activation.  

We can also calculate the contribution of each feature using their Shapley values, which can be seen in this plot. (shapley_values.png)

Interestingly, the 2 most important features from the importane plot are not even in the top 9 using this metric. The most important features are days_since_last_event, and observed_duration_days, which indicates that the timing and length of the journeys is more important than the other features and that we should focus on those in future models.

We can drop features with a low value, less than 0.01, and refit the model to see if there is an improvement.

This leads to a much simpler value that increases the accuracy ever so slightly. Notably, this model predicts the outcomes of the open journeys more accurately than any other model we have run, including the model predicting all unsuccesful journeys.

Some summary statistics were calculated for this model:
1. ROC AUC: 0.5
2. Log Loss: 8.43
3. Average Precision Score: 0.23
4. Confusion Matrix
    True Negatives: 70393
    False Positives: 0
    False Negatives: 21495
    True Positives: 2




# Task 4

I performed Cross-Validation on my training data set with 5 folds.

Fold 1: Weighted Accuracy = 0.9508
Fold 2: Weighted Accuracy = 0.9501
Fold 3: Weighted Accuracy = 0.9500
Fold 4: Weighted Accuracy = 0.9502
Fold 5: Weighted Accuracy = 0.9502



Contrary to normal situations, I think our training error actually has a chance to be higher than our testing error. This is becuase we have added the sample importance feature in the xgboost model, where an observation that has an ongoing journey of 100 days is 100 times more important than a journey that is just 1 day. While this sample importance helps us with the testing set, it is not true in the training set that we are 100 times more likely to be in a journey that is 100 days vs 1 day; since we sample from each observation once. As a result, we will be validating on our training data which is not representative of the changes we made to specifically target the testing data.

Additionally, I have set the threshold for predicting a successful user at 0.6 instead of 0.5. By knowing that most of our testing set is unsuccessful orders, we want our model to be certain that the observation is a successful order before predicting that outcome.

# Task 5

We submitted an intital Kaggle prediction by predicting an unsuccessful journey for every single test case. Since we know 95 percent of the observations are unsuccessful journeys, our accuracy could not be lower than 95 percent, providing a pretty good baseline for our other models. 

Later, we tested using our XGB models. My model predicted only about 500 successful users our of the 512,000 test cases. This is only about 0.1 percent of the testing set, meaning we are only correctly (assuming they were all correct which isn't true since it did worse than predicting all 0's) finding 2 percent of all the successful test cases. 