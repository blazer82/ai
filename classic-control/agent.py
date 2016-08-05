import numpy as np


class Agent:
	def __init__(self, env, model):
		self.env = env
		self.model = model

	def play(self):
		terminal = False
		observation = self.env.reset()
		X = np.zeros((2,) + observation.shape)
		X[0] = observation
		X[1] = observation

		total_reward = 0
		while terminal == False:
			y = self.model.predict(X)
			action = np.argmax(y)

			observation, reward, terminal, info = self.env.executeAction(action)
			total_reward += reward

			X[0] = X[1]
			X[1] = observation

		return total_reward

	def learn(self, overfit=False):
		terminal = False
		observation = self.env.reset()
		X = np.zeros((2,) + observation.shape)
		X[0] = observation
		X[1] = observation

		experience = []

		while terminal == False:
			y = self.model.predict(X)
			action = np.argmax(y)

			observation, reward, terminal, info = self.env.executeAction(action)

			experience.append((X.copy(), y.copy(), reward, terminal))

			X[0] = X[1]
			X[1] = observation

		nbr_experiences = len(experience)
		X_t = np.zeros((nbr_experiences,) + experience[0][0].shape)
		y_t = np.zeros((nbr_experiences,) + experience[0][1].shape)

		for i in reversed(range(0, nbr_experiences)):
			X, y, reward, terminal = experience[i]

			X_t[i] = X
			y_t[i] = y
			# print(y)

			action = np.argmax(y_t[i])
			y_t[i, action] = 1. -2. * .9**(nbr_experiences - (i+1))
			y_t[i] -= np.mean(y_t[i])

		while overfit:
			self.model.learn(X_t, y_t)

		self.model.learn(X_t[-300:], y_t[-300:])

		return nbr_experiences
