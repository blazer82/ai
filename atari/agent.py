import numpy as np
from memory import Memory


class Agent:
	def __init__(self, env, model, epsilon=.9, min_epsilon=.1, epsilon_decay=1e-3):
		self.env = env
		self.model = model
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon
		self.epsilon_decay = epsilon_decay
		self.episode = 0
		self.positiveMemory = Memory(model=self.model, episode_max_size=30)
		self.negativeMemory = Memory(model=self.model, episode_max_size=20)

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
			episode = []
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

					episode.append((X.copy(), y, action, reward, terminal))

					if reward == 1:
						self.positiveMemory.add(episode, positive=True)
						episode = []
					if reward == -1:
						self.negativeMemory.add(episode, positive=False)
						episode = []

					X[0] = X[1]
					X[1] = observation

		print "Score %f"%(total_reward / games)

		X_pos, y_pos = self.positiveMemory.sample(nbr_positive=games*10)
		X_neg, y_neg = self.negativeMemory.sample(nbr_negative=games*10)

		if not X_pos is None:
			print "Sample %d positive and %d negative memories"%(len(y_pos), len(y_neg))
			X_t = np.concatenate((X_pos, X_neg))
			y_t = np.concatenate((y_pos, y_neg))
		else:
			print "Sample %d negative memories"%(len(y_neg))
			X_t = X_neg
			y_t = y_neg

		while overfit:
			self.model.learn(X_t, y_t, nb_epoch=epochs)

		history = self.model.learn(X_t, y_t, nb_epoch=epochs)

		return total_reward / games, history.history['loss'][0], 0., epsilon
