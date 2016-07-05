import gym
import lutorpy as lua
import numpy as np

require('torch')
require('nn')


mlp = nn.Sequential()
mlp._add(nn.Linear(100800, 100))
mlp._add(nn.ReLU())
mlp._add(nn.Linear(100, 100))
mlp._add(nn.ReLU())
mlp._add(nn.Linear(100, 3))
mlp._add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()


env = gym.make('Breakout-v0')


for i_episode in range(4):
    observation = env.reset() # shape (210, 160, 3)
    for t in range(100):
        env.render()
        
        if t < 3:
            action = env.action_space.sample() - 1
            forward = torch.Tensor(1)
            forward[0] = action
        else:
            input = np.array(observation, dtype=float).flatten()
            x = torch.fromNumpyArray(input)
            forward = mlp._forward(x)
            action = np.argmax(forward.asNumpyArray())
        
        observation, reward, done, info = env.step(action)
        
        print "action was %f reward was %f"%(action, reward)
        
        
        if done:
            print("############")
            print("## FAILED ## Episode finished after {} timesteps".format(t+1))
            print("############")
            break
