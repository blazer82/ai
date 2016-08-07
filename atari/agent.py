import numpy as np


class Agent:
	def __init__(self, env, model, epsilon=.9, min_epsilon=.1, epsilon_decay=1e-3):
		self.env = env
		self.model = model
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decay = epsilon_decay
		self.episode = 0

	def play(self):
		terminal = False
		observation = self.env.reset()
		X = np.zeros((2,) + observation.shape)
		X[0] = observation
		X[1] = observation

		total_reward = 0
		while terminal == False and total_reward < 200:
			y = self.model.predict(X)
			action = np.argmax(y)

			observation, reward, terminal, info = self.env.executeAction(action)
			total_reward += reward

			X[0] = X[1]
			X[1] = observation

		return total_reward

	def learn(self, overfit=False, skip_frame=4):
		self.episode += 1.
		terminal = False
		observation = self.env.reset()
		X = np.zeros((2,) + observation.shape)
		X[0] = observation
		X[1] = observation

		epsilon = max(self.min_epsilon, self.epsilon - self.episode * self.epsilon_decay)

		experience = []
		total_reward = 0
		frame = 0
		action = self.env.env.action_space.sample()
		while terminal == False and total_reward < 200:
			frame += 1

			if frame%skip_frame != 0:
				observation, reward, terminal, info = self.env.executeAction(action)

			if frame%skip_frame == 0 or reward != 0 or terminal:
				y = self.model.predict(X)

				if frame%skip_frame == 0:
					if np.random.rand() <= epsilon:
						action = np.random.randint(0, len(y))
					else:
						action = np.argmax(y)

					observation, reward, terminal, info = self.env.executeAction(action)

				total_reward += reward

				experience.append((X.copy(), y, reward, terminal))

				X[0] = X[1]
				X[1] = observation

		nbr_experiences = len(experience)
		X_t = np.zeros((nbr_experiences,) + experience[0][0].shape)
		y_t = np.zeros((nbr_experiences,) + experience[0][1].shape)
		q = np.zeros(nbr_experiences)

		distance = 0
		mod = 0.
		for i in reversed(range(0, nbr_experiences)):
			X, y, reward, terminal = experience[i]

			X_t[i] = X
			y_t[i] = y
			q[i] = max(y)
			# print(y)

			if reward == 0:
				distance += 1
			else:
				distance = 0
				mod = reward

			mod *= .99**distance

			action = np.argmax(y_t[i])
			y_t[i, action] = mod


		while overfit:
			self.model.learn(X_t, y_t)

		history = self.model.learn(X_t, y_t)

		return total_reward, history.history['loss'][0], np.mean(q), epsilon
