import numpy as np
import gym
from PIL import Image


class Environment:
	TYPE_PONG='Pong-v0'

	def __init__(self, env_type):
		self.env = gym.make(env_type)
		self.must_be_reset = True

	def preprocessObservation(self, observation):
		grey = np.average(observation, 2)
		img = Image.fromarray(grey)
		img.thumbnail((110, 110), Image.NEAREST) # new shape (110, 84)
		img = img.crop((5, 25, 79, 105)) # new shape (80, 74)
		x_new = np.asarray(img, dtype=np.float32).copy()
		x_new -= np.mean(x_new)
		return x_new

	def reset(self):
		self.must_be_reset = False
		return self.preprocessObservation(self.env.reset())

	def executeAction(self, action):
		if self.must_be_reset:
			self.reset()

		observation, reward, terminal, info = self.env.step(action)

		if terminal:
			self.must_be_reset = True

		return self.preprocessObservation(observation), reward, terminal, info
