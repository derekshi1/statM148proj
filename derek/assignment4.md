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

# Task 4 - Derek

I tried using Temporal Point Process (TPP) because it is designed to predict the timing and nature of discrete events within a continuous time window, differing from RNN and LSTM. TPP treats the journey as a dynamic intensity, modeling the probability of a success event occurring given the user's history and the time elapsed since their last action.To make this work with our specific dataset, we transformed raw customer milestones into sequences of event IDs and log-scaled time deltas. We shifted the targets: at any point $i$ in a journey, the model is trained to predict the event type and the specific time gap for step $i+1$. We designed it this way with the "real-time" requirement in mind in the testing set. Because we model the decay of intent over time, the model can naturally distinguish between a "hot" lead who is actively clicking and a "cold" lead who has been idle for days. 

A primary consideration was the 95/5 class imbalance in testing, which we addressed using weighted cross-entropy and sampling to ensure the model prioritizes the rare "Order Shipped" signal. We also accounted for the fact that journeys in the test set are unfinished by utilizing the TPP's survival function, which allows us to calculate the probability of a future shipment without requiring the journey to be artificially truncated.

There were also many considerations in terms of where we would pad our sequences (start end?), and the max_len we would pad our sequences to. We are working on having this model train successfully, and will test it in the the next weeks hopefully!
