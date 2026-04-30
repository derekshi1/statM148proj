# Assignment 4: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay


# Task 1 
# Task 2 - 
Got a new accuracy of 0.04685 by using an ensemble model with XGBoost, Logistic Regression, LSTM, and a Neural Network.  The Neural Network and the XGBoost probabilities had a correlation of about 99%, so we can continue to explore other model architectures like transformers, CNN, and GRU.# Task 3 - Caoimhe

# Task 4 - Derek

I tried using Temporal Point Process (TPP) because it is designed to predict the timing and nature of discrete events within a continuous time window, differing from RNN and LSTM. TPP treats the journey as a dynamic intensity, modeling the probability of a success event occurring given the user's history and the time elapsed since their last action.To make this work with our specific dataset, we transformed raw customer milestones into sequences of event IDs and log-scaled time deltas. We engineered the model to be self-supervised by shifting the targets: at any point $i$ in a journey, the model is trained to predict the event type and the specific time gap for step $i+1$. We designed it this way with the "real-time" requirement in mind in the testing set. Because we model the decay of intent over time, the model can naturally distinguish between a "hot" lead who is actively clicking and a "cold" lead who has been idle for days. A primary consideration was the 95/5 class imbalance in testing, which we addressed using weighted cross-entropy and sampling to ensure the model prioritizes the rare "Order Shipped" signal. We also accounted for the fact that journeys in the test set are unfinished by utilizing the TPP's survival function, which allows us to calculate the probability of a future shipment without requiring the journey to be artificially truncated.


We are working on having this model train successfully, and will test it in the coming weeks.
