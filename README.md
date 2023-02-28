Note that the data is downloaded from Niv's lab open source:
https://nivlab.princeton.edu/data

The original data was saved as .mat format and we tranformed it into .csv format to facilitate python modeling. Here we also paste the readme of the data varialbe. 

<blockquote>
Data from: Y Niv, R Daniel, A Geana, SJ Gershman, YC Leong, A Radulescu & RC Wilson (2015) - Reinforcement learning in multidimensional environments relies on attention mechanisms - J Neuroscience 35(21): 8145-8157; doi:10.1523/JNEUROSCI.2978-14.2015

please cite appropriately if used in publications

Subjects ran 500 (fast) trials with signaled changes, (optional, for some subjects only: 500 fast trials of unsignaled changes), and then 300 (slower) trials of signaled changes while we acquired fMRI. This file DimTaskData includes all data for 800 signaled changes trials (500 + 300) for the 22 subjects analyzed in the paper, in the following format:

Choices - choices (800 trials x 3 features x 22 subjects), NaN for missed trials 
Outcomes - outcomes, 1 or 0 (800 trials x 22 subjects), NaN for missed trials
Stimuli - stimuli (800 trials x 3 dimensions x 3 features x 22 subjects; the first dimension is colors, then shapes, then textures)
RelevantDim - relevant dimension in each trial (800 trials x 22 subjects) 
CorrectFeature - correct feature in each trial (800 trials x 22 subjects)
ReactionTimes - reaction times (800 trials x 22 subjects), NaN for missed trials

Note: Trials were "missed" if the subject was too slow to respond. In this case many of the vectors/matrices above have a NaN so be sure to deal with that properly in your code
</blockquote>


