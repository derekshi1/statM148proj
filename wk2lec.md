 in the training, all the journeys are completed as unsuccessful or successful

 but in the testing, all the journeys are ongoing. 

 so the issue is that if we train something on all the completed journeys, there will be indicators that won't show up in the testing data so then we have big issues and will not predict anything correctly


**SOO what we should do is uniformly sample from a random time in a journey and then truncate there in the TRAINING DATA, first possible cut is the first event, unsuccessful: the last cut possible is between before the end of the 60 days for inactivity, successful: first cut is the first order, then the success**

^^issue with cutting at an event, there is always a recent event, but one of the big things is days of inactivity to predict for not successful journeys, so cutting at a time solves that


**After 60 days of inactivity it is automatically unsuccessful, so if there is an ongoing journey at liek 59 days then it is likely going to be unsuccessful**

**also can consider the time of the day, like in terms of christmas etc- LOOK INTO THIS**

**IN THE TEST DATA: we are more likely to be in long journeys than short ones, thus more likely to cut into unsuccessful journeys because they are longer, the ideas is just to sample from the training for longer journeys MORE, if a journey is 1 day sample once, if a journey is 2 days sample twice, if a journey is N days sample N times**

TLDR: in the training sample time uniformly across journeys, if the joruney is one day sample once, if a journey is 5 days sample 5 times, if a journye is N days ample N times. 

secondary: keep in mind time of day in journeys leading up to "now", like it should be an input in whatever model we use


in test set it is 5% success (longer journeys), in training set it is 20% success



