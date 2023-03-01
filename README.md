Note that the data is downloaded from Niv's lab open source:
https://nivlab.princeton.edu/data

The original data was saved as .mat format and we tranformed it into .csv format to facilitate python modeling. Saved as "data/raw_data.csv".

We also redefine some variables to fit in the dataframe. 

* sub_id: the subject id
* s0: the configuration of stimulus 1. The configuration is represented as a 3-number string. From left to right, each number indicates the feature in the ith dimension. 
* s1: the configuration of stimulus 2. 
* s3: the configuration of stimulus 3.
* relevantDim: the relevant dimension. The dimension that correspond to the reward.
* correcPhi: the correct feature. The feature that correspond to the reward. 
* choice: the selected stimulus configuration, which is inherited from the original data 
* act: action. The selected stimulus number. 0-s0, 1-s1, 2-s2.
* rew: reward. The outcome/received reward after participants' response. 




