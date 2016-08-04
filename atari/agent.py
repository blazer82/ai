import numpy as np


class Agent:
	def __init__(self, env, model):
		self.env = env
		self.model = model

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

			experience.append((X, y, reward, terminal))

			X[0] = X[1]
			X[1] = observation

		nbr_experiences = len(experience)
		X_t = np.zeros((nbr_experiences,) + experience[0][0].shape)
		y_t = np.zeros((nbr_experiences,) + experience[0][1].shape)
		for i in range(0, nbr_experiences - 2):
			X, y, reward, terminal = experience[i]

			X_t[i] = X
			y_t[i] = y

			action = np.argmax(y_t[i])
			q_next = np.max(experience[i + 1][1])

			y_t[i][action] = (1. - terminal) * .99 * q_next + reward

		while overfit:
			self.model.learn(X_t, y_t)

		self.model.learn(X_t, y_t)
