import numpy as np
import math
import random

import torch


class EpsilonDecay():
    def __init__(self,e_start,e_end,decay):
        self.e_start = e_start
        self.e_end = e_end
        self.decay = decay
        self.step = 0        #check

    def select_action(self,state,action_number,q_net):
        self.step = self.step + 1
        epsilon = self.e_end + (self.e_start - self.e_end) * math.exp(-1 * self.step / self.decay)
        rand_num = random.random()
        if rand_num<=epsilon:
            #select the exploration
            action=random.randrange(action_number)
            return torch.as_tensor(action)
        else:
            #select exploitation
            action= q_net.forward(state.unsqueeze(0))
            action=action.squeeze(0)
            action=action.max(0)[1]
            return action
