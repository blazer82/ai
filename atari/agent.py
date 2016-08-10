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

	def learn(self, overfit=False, games=1, epochs=1, skip_frame=2):
		self.episode += 1.
		epsilon = max(self.min_epsilon, self.epsilon - self.episode * self.epsilon_decay)

		experience = []
		total_reward = 0

		for game in range(1, games + 1):
			print "Game %d/%d..."%(game, games)
			terminal = False
			observation = self.env.reset()
			X = np.zeros((2,) + observation.shape)
			X[0] = observation
			X[1] = observation
			frame = 0
			action = np.random.randint(0, 2)
			while terminal == False:
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

					experience.append((X.copy(), y, action, reward, terminal))

					X[0] = X[1]
					X[1] = observation

		nbr_experiences = len(experience)
		X_t = np.zeros((nbr_experiences,) + experience[0][0].shape)
		y_t = np.zeros((nbr_experiences,) + experience[0][1].shape)
		q = np.zeros(nbr_experiences)

		mod = 0.
		for i in reversed(range(0, nbr_experiences)):
			X, y, action, reward, terminal = experience[i]

			X_t[i] = X
			y_t[i] = y
			q[i] = max(y)

			if reward != 0:
				mod = reward + 0.
			else:
				mod *= .99

			y_t[i, action] = mod


		while overfit:
			self.model.learn(X_t, y_t, nb_epoch=epochs)

		history = self.model.learn(X_t, y_t, nb_epoch=epochs)

		return total_reward / games, history.history['loss'][0], np.mean(q), epsilon
