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