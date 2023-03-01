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
        v_stims = [self.V[s] for s in self.s_embed(stims)]
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

# ---------- RL models ----------- #

class fRL(naiveRL):
    '''The feature RL'''
    name  = 'feature RL'
    pname = ['η', 'β']

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
        self._init_W()

    def _init_W(self):
        self.W = np.zeros([self.nD*self.nF])

    # -------- embeddings --------- #

    def s_embed(self, s):
        return [to_phi(i) for i in s]
    
    # --------- decision --------- #

    def policy(self, stims):

        # Softmax(βV(s_i)): V(s_i) = ∑_fW(f)
        v_stims = [(f*self.W).sum() 
                   for f in self.s_embed(stims)]
        return softmax(self.beta*v_stims)
    
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

        # retrieve memory 
        ss, a = self.mem.sample('s', 'a')
        f_chosen = self.s_phi(ss)[a]

        # W(f) = (1-d)W(f)  ∀f is not chosen
        f_unchosen = 1 - f_chosen
        self.W -= self.d*self.W*f_unchosen

# -------- Bayesian model ----------- #

class bayes(fRL):
    '''The bayesian learning model'''
    name  = 'bayesian learning'
    pname = ['β']

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
        self._init_p_F()

    def _init_p_F(self):
        # The probability of each feature being 
        # the target feature is initialized at 1/9 
        # at the beginning of a game 
        n = self.nD*self.nF
        self.p_F = np.ones([n]) / n

    # --------- decision --------- #

    def policy(self, stims):
        
        # construct V(s)
        ss = self.s_phi(stims)
        for s in ss:
            f = (np.eye(self.nD*self.nF)@s.reshape([-1, 1])).reshape([-1])

        # Softmax(βV(s_i)): V(s_i) = ∑_fW(f)
        v_stims = [(f*self.W).sum() 
                   for f in self.s_embed(stims)]
        return softmax(self.beta*v_stims)
    
    # --------- learning --------- #

    def update(self):
        self.update_Bel()

    def update_Bel(self):
        
        # Retrieve memory 
        ss, a, r = self.mem.sample('s', 'a', 'r')
        f_chosen = (self.s_phi(ss)[a]).reshape([-1, 1])
        
        # update the belief according to Bayes' rule
        # p(f) ∝ p(R|f, c)p(f)
        # the first term is p=.75 or p=.25
        # depending on the reward on the current
        # whether the current c included f
        # p=.75 if R=1 given current c included in f
        # p=.25 if R=0 given current c included in f
        # p=0   else
        p_R1FC = (.25+.5*r*np.eye(self.nD*self.nF)@f_chosen).reshape([-1])
        f_F    = p_R1FC*self.p_F
        self.p_F = f_F / f_F.sum() 
        
        



    
    


    

    
    



if __name__ == '__main__':

    print(1)

    


        
    



