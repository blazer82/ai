import gym
import lutorpy as lua
import numpy as np

require('torch')
require('nn')

mlp = nn.Sequential()

mlp._add(nn.Linear(4, 4))
mlp._add(nn.Tanh())
mlp._add(nn.Linear(4, 4))
mlp._add(nn.Tanh())
mlp._add(nn.Linear(4, 4))
mlp._add(nn.Tanh())

mlp._add(nn.Linear(4, 2))
mlp._add(nn.LogSoftMax())

criterion = nn.MSECriterion()
#trainer = nn.StochasticGradient(mlp, criterion)
#trainer.learningRate = .2

env = gym.make('CartPole-v0')
env.reset()

for i_episode in range(1000):
    observation = env.reset()
    for t in range(200):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        x = torch.fromNumpyArray(observation)

        output = mlp._forward(x)
        action = np.argmax(output.asNumpyArray())
        
        
        #print(action)
        observation, reward, done, info = env.step(action)
        
        if done:
            reward = 0.
            
        targets = torch.Tensor(2)._zero()
            
        if reward < 1:
            targets[action] = max(-1., -.1*t)
        else:
            targets[action] = min(1., .01*t)
        
        print "Action #%d: %d"%(t, action)
        print(output.asNumpyArray())
        print(reward)
        print(targets.asNumpyArray())
        
        loss = criterion._forward(output, targets)
        mlp._zeroGradParameters()
        mlp._backward(x, criterion._backward(output, targets))
        mlp._updateParameters(.01)
        if done:
            print("############")
            print("## FAILED ## Episode finished after {} timesteps".format(t+1))
            print("############")
            break