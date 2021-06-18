import torch
import torch.nn as nn
import gym
import numpy as np
from torch.optim import Adam
from NeuralNet import Network
from ReplayMemory import ReplayMemory
from EpsilonDecay import EpsilonDecay
from dqn_wrapper import SkipFrame
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


def preprocess(env_, stack_frame_no, resize_size, skip_number):
    env_new = SkipFrame(env_, skip_number)
    env_new = GrayScaleObservation(env_new)
    env_new = ResizeObservation(env_new, shape=resize_size)
    env_new = FrameStack(env_new, stack_frame_no)
    return env_new


def state_process(state_):
    state_new = state_.__array__()
    state_new = state_new.squeeze(axis=None)
    state_new = torch.Tensor(state_new / 255)
    return state_new


def optimize_step(batch_size_, gamma_, loss_fn_, optimizer_):
    if len(memory.replay_memory) < batch_size_:
        return
    memory_batch = memory.sample(batch_size_)
    state_list = [memory_element[0] for memory_element in memory_batch]
    states = torch.stack(state_list)
    new_state_list = [memory_element[3] for memory_element in memory_batch]
    new_states = torch.stack(new_state_list)
    actions_list = [memory_element[1] for memory_element in memory_batch]
    actions = torch.stack(actions_list)
    reward_list = [memory_element[2] for memory_element in memory_batch]
    rewards = torch.stack(reward_list)
    target = target_net(new_states).max(dim=1)[0]
    q_val = q_net(states)[np.arange(0, batch_size_), actions]
    target = rewards + gamma_ * target
    loss = loss_fn_(q_val, target)
    loss.backward()
    optimizer_.step()
    optimizer_.zero_grad()


if __name__ == '__main__':
    env_name = "SpaceInvaders-v0"
    env = gym.make(env_name)
    env= preprocess(env, 4, 84, 4)
    action_number = env.action_space.n
    input_image_channel_number = 4
    lr = .001
    epsilon_start = 0.9
    epsilon_end = 0.05
    decay = 0.1
    memory_limit = 1000
    batch_size = 32
    gamma = 0.999
    target_update = 20
    q_net = Network(in_size=input_image_channel_number, out_size=action_number)
    target_net = Network(in_size=input_image_channel_number, out_size=action_number)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = Adam(q_net.parameters(), lr)
    loss_fn = nn.SmoothL1Loss()
    memory = ReplayMemory(memory_size=memory_limit)
    epsilon_decay = EpsilonDecay(e_start=epsilon_start, e_end=epsilon_end, decay=decay)
    episodes = 100
    for i in range(episodes):
        state = env.reset()
        state = state_process(state)
        ep_rew = 0
        time_step = 0
        while True:
            time_step += 1
            if time_step % 20 == 0:
                env.render()
            action = epsilon_decay.select_action(state, action_number, q_net)
            new_state, reward, done, _ = env.step(action)
            ep_rew += reward
            new_state = state_process(new_state)
            temp = (state, action, torch.as_tensor(reward), new_state)
            memory.insert_transition(temp)
            state = new_state
            optimize_step(batch_size, gamma, loss_fn,optimizer)

            if done:
                print("Episode no {0:f} , reward ".format(i))
                print(ep_rew)
                break
        if i % 20 == 0:
            target_net.load_state_dict(q_net.state_dict())
