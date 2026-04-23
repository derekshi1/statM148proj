# statM148proj
stat m148 final proj


so there are different "journeys", unsuccessful if 60 days in between, successful if less tahtn 60 days and order shipped, ongoing has not termination and hasn't gone over 60 days.


Success: Order shipped
On-Going: An action has been taken before 60 days from the endpoint (less than 60 days from the cutoff)
Unsuccessful: no action has been taken 60 days from the endpoint

Endpoint: The last timestamp in the training data

Interested in behavior of on-going journeys 
-> what kind of interventions can we do in order to convert them to success?
-> what are the current probabilities of a given on-going journey being successful?

For successful journeys, we assume that there are a bunch of actions leading up to the end of the journey that are associated with it being successful. However, everything before these unique actions occur are probably not


Presentation:
- contrasting Cp Profiles with GAM models


numerical, categorical
teaching
applied example to our data specifically
logistic regression CP plot vs random forest CP plot
specifically a slide on why it is important, why do we care? 
why do we can about interrpreting black box machine learning models? 

## Ice Plots and PDP Plots
- Individual conditional expectation (ICE) plots
    - displays how the prediction of every instance, or observation, changes when a particular feature changes.
    - This is like a CP plot, but for every instance.
- Partial Independece Plot (PDP) plot
    - averages the model prediction of an ice plot. summarizes overal effect of predictor
    - PDP may evaluate unrealistic feature combinations when predictions are correlated
- PDP and ICE are only reliable around the local neighborhood fof the true observations

## Shapley Values
- Method for faily distributing total gains or cost among a group of players who have collaborated
- Machine learning for feature importance
- Game Theory Example:
    - how do we split gold fairly amongst three players
    - A: average marginal contribution for a particular player
    - Shapley Value: Add the marginal gain for a particular player across number of appearances. Then divide by the number of appearances
- Results: useless features has a 0 value
    - if explanatory variables are equal, they have the same shapley value
- Visualization: Beeswarm plot shows magnitude AND prediction
    - Probably better to start with a normal feature importance plot then create a separate scatter plot for the top important features to see the direction
- Pros: solid theoretical game theory, accounts for all possible subset
- Con: Computational cost-2^p. Approximation can be done monte carlo