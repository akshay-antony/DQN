import gym
import random
import time
import numpy as np
import torch
import torchvision.transforms as tt
from dqn_wrapper import SkipFrame
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
import matplotlib.pyplot as plt
env = gym.make('SpaceInvaders-v0')
env=SkipFrame(env=env,skip_number=4)
env=GrayScaleObservation(env)
env=ResizeObservation(env,shape=84)
env = FrameStack(env,num_stack=4)

print(env.observation_space.shape)

env.reset()
t=0
transform=tt.ToTensor()

# def get_screen(env):
#     screen=env.render(mode='rgb_array').transpose(2,0,1)
#     _,hieght,width=screen.shape
#     #print(hieght,width,screen.shape)
#     #plt.imshow(screen[0])
#     #plt.show()
#
#
for _ in range(10000):
    while True:
        t+=1
        if t % 5000 == 0:
            # env.render()
             print(obs[0])
            # print(obs[0].shape)
            # print(obs.shape)
            # figure=plt.figure(figsize=(8,8))
             row=2
            # column=2
            # figure.add_subplot(row,column,1)
            # plt.imshow(obs[0])
            # plt.show()
            # figure.add_subplot(row, column,2)
            # plt.imshow(obs[1])
            # plt.show()
            # figure.add_subplot(row, column, 3)
            # plt.imshow(obs[2])
            # plt.show()
            # figure.add_subplot(row, column, 4)
            # plt.imshow(obs[3])
            # plt.show()
        obs,rew,done,info=env.step(env.action_space.sample())  # take a random action
        obs=obs.__array__()
        #print(obs.shape)
        obs=obs.squeeze(axis=None)
        print(obs.shape/255)
        obs=torch.Tensor(obs)
        obs=ta(obs)
        if done:
            env.reset()
            break
env.close()
