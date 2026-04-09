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

