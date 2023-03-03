This repo replicates some results in 

> Niv, Y., Daniel, R., Geana, A., Gershman, S. J., Leong, Y. C., Radulescu, A., & Wilson, R. C. (2015). Reinforcement learning in multidimensional environments relies on attention mechanisms. Journal of Neuroscience, 35(21), 8145-8157.

Note that the data is downloaded from Niv's lab open source data:
https://nivlab.princeton.edu/data

The original data was saved as .mat format and we tranformed it into .csv format to facilitate python modeling. Saved as "data/raw_data.csv".

We also redefined some variables to fit in the dataframe. 

* sub_id: the subject id.
* s0: the configuration of stimulus 1. The configuration is represented as a 3-number string. From left to right, each number indicates the feature in the ith dimension. 
* s1: the configuration of stimulus 2. 
* s3: the configuration of stimulus 3.
* relevantDim: the relevant dimension. The dimension that correspond to the reward.
* correcPhi: the correct feature. The feature that correspond to the reward. 
* choice: the selected stimulus configuration, which is inherited from the original data.
* act: action. The selected stimulus index. 0-s0, 1-s1, 2-s2.
* rew: reward. The outcome/received reward after participants' response. 
* trial: the number of a trial within a game.
* game: the number of a game per participant.

The data is called "raw" but not literally is. We conducted a few preprocessings, including some nan value replacement. Please use the data from Niv's lab for a REAL research. 

Run ```python m1_main.py``` to start. 
