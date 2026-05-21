# Assignment7: Group 4
## Derek Shi, Caoimhe (Queeva) Curran, Jayden Tani, Emilio Dulay

# Task 1 - Caoimhe

# Task 2 - 

# Task 3 - Derek

Using my XGBoost model, I was able to model the success probability of 2 specific journeys, one resulting in a success and one resulting in a failure. To accurately do this, I had to find the features which were time dependent (e.g. time since last event) and adjust them to match the inactivity or activity seen throughout the journey, to then create predictions. Otherwise, there would have been 0 change in predicted probability between events due to lack of inactivity since the features reflecting that would be static.


![The joint probability for a successful journey](joint_probability_timeline_successful.png)
![The joint probability for an unsuccessful journey](joint_probability_timeline_unsuccessful.png)


# Task 4 - 