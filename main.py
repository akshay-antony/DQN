import torch
from torch.optim import Adam
from NeuralNet import Network
from ReplayMemory import ReplayMemory
from EpsilonDecay import EpsilonDecay
import gym
import torchvision.transforms as tt
def transform_input(image):
    transform_image=tt.Compose(tt.ToPILImage(),
                               tt.Grayscale(num_output_channels=1),
                               tt.Resize(size=(84,84))
                               tt.ToTensor()
                               )
if __name__ == '__main__':
        env_name="CartPole-v0"
        env=gym.make_env(env_name)
        action_number=env.action_space.shape.n
        input_image_channel_number=4
        lr=.001
        epsilon_start=0.99
        epsilon_end=0.01
        decay=0.1
        memory_limit=1000
        q_net=Network(in_size=input_image_channel_number,out_size=action_number)
        target_net=Network(in_size=input_image_channel_number,out_size=action_number)
        target_net.load_state_dict(q_net.state_dict())
        optimizer=Adam(q_net.parameters(),lr)
        memory=ReplayMemory(memory_size=memory_limit)
        epsilon_decay=EpsilonDecay(e_start=epsilon_start,e_end=epsilon_end,decay=decay)
        episodes=100
        while e in range(episodes):
            state=env.reset()
            while True:
                action=epsilon_decay.get_action(state,action_number,q_net)
                new_state,reward,_,_=env.step(action)
                temp=(state,action,reward,new_state)
                
