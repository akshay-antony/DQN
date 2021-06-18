import gym

class SkipFrame(gym.Wrapper):
    def __init__(self,env,skip_number):
        super().__init__(env)
        self.skip=skip_number
        self.env=env

    def step(self, action):
        total_reward=0
        for i in range(self.skip):
            obs,rew,done,info=self.env.step(action)
            total_reward+=rew
            if done:
                break
        return obs,total_reward, done, info


