'''Replicate some results in Niv 2015
@ZF
'''

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.special import softmax

# self-defined visualization
from utils.viz import viz
viz.get_style()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------- #
#             Functions               #
# ----------------------------------- #

to_idx = lambda x: np.sum([3**j*(int(s)-1) 
            for j, s in enumerate(reversed(x))])

to_phi = lambda x, nF: np.hstack([np.eye(nF
            )[int(s)-1] for s in x])

# ----------------------------------- #
#                Models               #
# ----------------------------------- #

class simpleBuffer:

    def __init__(self):
        self.keys = ['s', 'a', 'r']
        self.reset()

    def push(self, m_dict):
        '''Add a sample trajectory'''
        for k in m_dict.keys():
            self.m[k].append(m_dict[k])

    def sample(self, *args):
        '''Sample a trajectory'''
        lst = [self.m[k] for k in args]
        if len(lst) == 1: return lst[0]
        else: return lst 

    def reset(self):
        '''Empty the cached trajectory'''
        self.m = {k: [] for k in self.keys}

class baseAgent:
    '''The base agent'''

    def __init__(self, nD, nF, params):
        self.nD = nD
        self.nF = nF
        self.nS = nD**nF
        self._load_params(params)
        self._init_buffer()

    def _init_buffer(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        raise NotImplementedError

    def get_act(self, state, rng):
        return rng.choice(self.nA)

    def update(self):
        raise NotImplementedError
    
class naiveRL(baseAgent):
    '''The naive RL'''
    name  = 'naive RL'
    pname = ['η', 'β']

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)

    def _load_params(self, params):
        self.eta  = params[0]
        self.beta = params[1]

    def _init_V(self):
        self.V = np.zeros([self.nS])

    # -------- embeddings --------- #

    def s_embed(self, s):
        return [to_idx(i) for i in s]
    
    # --------- decision --------- #

    def policy(self, stims):

        # Softmax(βV(s_i))
        v_stims = [self.V[s] for s in stims]
        return softmax(self.beta*v_stims)
    
    # --------- learning --------- #

    def update(self):
        self.update_V()

    def update_V(self):

        # get data 
        ss, a, r = self.mem.sample('s', 'a', 'r')
        s_chosen = self.s_embed(ss)[a]

        # update: V(s_chosen) = V(s_chosen) + η(r-V(s_chosen))
        rpe = r - self.V[s_chosen]
        self.V[s_chosen] += self.eta*rpe 

class fRL(naiveRL):
    '''The feature RL'''
    name  = 'feature RL'
    pname = ['η', 'β']

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)

    def _init_W(self):
        self.W = np.zeros([self.nD*self.nF])

    # -------- embeddings --------- #

    def s_embed(self, s):
        return [to_phi(i) for i in s]
    
    # --------- learning --------- #

    def update_V(self):

        # get data 
        ss, a, r = self.mem.sample('s', 'a', 'r')
        f_chosen = self.s_phi(ss)[a]

        # cal value: V = W.T@f
        V_chosen = (f_chosen*self.W).sum()

        # update: W(f_chosen) = W(f_chosen) + η(r-V)
        rpe = r - V_chosen
        self.W += self.eta*rpe*f_chosen 

class fRL_decay(fRL):
    '''The feature RL with decay'''
    name  = 'feature RL'
    pname = ['η', 'd', 'β']

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)

    def _load_params(self, params):
        self.eta  = params[0]
        self.d    = params[1]
        self.beta = params[2]

    def update(self):
        self.decay()
        self.update_V()
        
    def decay(self):
        self.W -= self.d*self.W 




if __name__ == '__main__':

    print(1)

    


        
    



