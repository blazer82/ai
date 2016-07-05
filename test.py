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

criterion = nn.ClassNLLCriterion()
#trainer = nn.StochasticGradient(mlp, criterion)
#trainer.learningRate = .2

env = gym.make('CartPole-v0')
env.reset()

for i_episode in range(1000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        x = torch.fromNumpyArray(observation)
        forward = mlp._forward(x)
        action = np.argmax(forward.asNumpyArray())
        #print(action)
        observation, reward, done, info = env.step(action)
        if done or t < 5:
            reward = 0.
        #print(reward)
        y = torch.Tensor(1)
        if reward:
            y[0] = action
            rate = .001
        else:
            y[0] = (action + 1) % 2
            rate = min(.0005 * t, .002)
        print "action was %f reward was %f, y is %f"%(action, reward, y[0])
        y[0] += 1
        criterion._forward(forward, y)
        mlp._zeroGradParameters()
        mlp._backward(x, criterion._backward(mlp.output, y))
        mlp._updateParameters(rate)
        if done:
            print("############")
            print("## FAILED ## Episode finished after {} timesteps".format(t+1))
            print("############")
            break