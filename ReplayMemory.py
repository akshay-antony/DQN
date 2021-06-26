from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np

class ReplayMemory():

    def __init__(self,memory_size):
        self.replay_memory=deque(maxlen=memory_size)

    def insert_transition(self,transition):
        # transition state action next_state reward#
        self.replay_memory.append(transition)

    def sample(self,batch_size):
        return random.sample(self.replay_memory, batch_size)


