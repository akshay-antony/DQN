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


# pr=ReplayMemory(20)
# for i in range(10):
#     temp=torch.rand(2)
#     temp2=torch.as_tensor(random.randint(0,2))
#     temp3=(temp,temp2)
#     pr.insert_transition(temp3)
# memory=pr.sample(5)
# # # s=torch.cat(tem[0] for tem in memory)
# # # print(s)
# # # #print(memory[0][0])
# # # s=torch.cat([tem[0] for tem in memory])
# state= [tem[0] for tem in memory]
# action=[tem[1] for tem in memory]
# #torch.as_tensor(action)
# q=torch.stack(action)
# print(q[0])
# # # state2=torch.as_tensor(state)
# # # print(s)
# # # print(state[0])
# p=torch.stack(state)
# q=torch.stack(action)
# p=torch.as_tensor(p)
# q=torch.as_tensor(q)
# q=q.unsqueeze(dim=-1)
# print(p.size(),q.size())
# nnn=p.gather(dim=-1,index=q)
# print(p)
# print(q)
# print(nnn)
# print(p.size(),q.size())
# # print(p[1])
#
# net=nn.Sequential(nn.Flatten(),
#                   nn.Linear(2,10),
#                   nn.ReLU(),
#                   nn.Linear(10,2),
#                   nn.ReLU())
# testin_max=torch.rand(32,2,dtype=torch.float32)
#target=np.random.randint(2,size=(32))
# target=[torch.as_tensor(random.randint(0,1)) for i in range(32)]
# target=torch.stack(target)
# p=torch.rand((32,2,1),requires_grad=False)
# #print(net(p).max(dim=1)[0])
# # print(target.size())
# print(net(p))
# print(target)
# loi=net(p)[np.arange(0,32),target]
# print(loi)
# target2=[random.randint(0,1) for i in range(32)]
# target2=torch.as_tensor(target2).unsqueeze(-1)
# print(target2.size())
# print(testin_max)
# print(target2)
# sm=testin_max.gather(dim=-1,index=target2)
# asa=testin_max.max(1)[0]
# print(sm)
# asa2=torch.max(testin_max,dim=1)[0]
# #print(asa)
# #print(asa2)