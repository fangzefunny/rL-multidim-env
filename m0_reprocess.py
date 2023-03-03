'''Replicate some results in Niv 2015
@ZF
'''

import os 
import numpy as np 
import pandas as pd
from scipy.io import loadmat

pth = os.path.dirname(os.path.abspath(__file__))

# --------  Functions -------- #

cate  = lambda x: ''.join(list(x))
match = lambda x: np.random.choice(3) if x['choice']=='000' else [
        x[f's{s}'] for s in range(3)].index(x['choice']) 

def to_csv(mat):

    nSub = mat['DimTaskData'][0][0][1].shape[-1]
    nRow = mat['DimTaskData'][0][0][1].shape[0]

    data = []

    for i in range(nSub):

        sub_data = {} 

        # get sub_id
        sub_data['sub_id'] = [i]*nRow
        # get stimuli
        for s in range(3):
            stims = list(mat['DimTaskData'][0][0][2][:, s, :, i].astype(str))
            sub_data[f's{s}'] = list(map(cate, stims))
    
        # get relevant dim:
        sub_data['relevantDim'] = mat['DimTaskData'][0][0][3][:, i]
        # get correct feature:
        sub_data['correctPhi']  = mat['DimTaskData'][0][0][4][:, i]
        
        # construct a dataframe 
        sub_data = pd.DataFrame.from_dict(sub_data)
        # get choice: replace nan with a random number  
        sub_data['choice'] = list(map(cate, list(np.nan_to_num(
                        mat['DimTaskData'][0][0][0][:, :, i],
                        copy=True, nan=0).astype(int).astype(str))))
        sub_data['act'] = sub_data.apply(match, axis=1)
        # get rew: replace nan with a random number
        sub_data['rew'] =  np.nan_to_num(mat[
            'DimTaskData'][0][0][1][:, i], nan=np.random.choice(1))

        # add to the data 
        data.append(sub_data)

    data = pd.concat(data, axis=0, ignore_index=True)

    # add trials and games columns to faciliate modeling
    sub_lst = data['sub_id'].unique() 
    new_data = pd.DataFrame(np.zeros([data.shape[0], 2]), 
                            columns=['trial', 'game'])
    r = 0
    for sub_id in sub_lst:
        sub_data = data.query(f'sub_id=={sub_id}').reset_index()
        t, g = -1, 0 
        prev_config = [sub_data.loc[0, 'relevantDim'], 
                       sub_data.loc[0, 'correctPhi']]
        for i, row in sub_data.iterrows():
            config = [row['relevantDim'], row['correctPhi']]
            if config == prev_config:
                t += 1
            else:
                t = 0
                g += 1
            new_data.loc[r, 'trial'] = t
            new_data.loc[r, 'game']  = g
            prev_config = [row['relevantDim'], row['correctPhi']]
            r += 1

    # pickle the data 
    con_data = pd.concat([data, new_data], axis=1)
    covert_dict = {k: 'int' for k in ['s0', 's1', 's2', 'choice', 'trial', 'game']}
    con_data = con_data.astype(covert_dict)
    con_data.to_csv(f'{pth}/data/raw_data.csv')
    
if __name__ == '__main__':

    to_csv(loadmat('data/BehavioralDataOnline.mat'))
    
    
    

