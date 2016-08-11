import numpy as np


class Memory:
	def __init__(self, model, size=1000):
		self.size = size
		self.model = model
		self.memory = []

	def add(self, episode, positive):
		if len(self.memory) >= self.size:
			self.memory = self.memory[1:]
		self.memory.append([np.asarray(episode), positive])

	def sample(self, nbr_positive=0, nbr_negative=0):
		pos = 0
		neg = 0
		len_memory = len(self.memory)
		total = min(nbr_negative + nbr_positive, len_memory)
		X_t = []
		y_t = []
		if total == 0: return None, None
		for i, idx in enumerate(np.random.randint(0, len_memory, size=total)):
			episode = self.memory[idx][0]
			positive = self.memory[idx][1]

			if positive and pos >= nbr_positive: continue
			if not positive and neg >= nbr_negative: continue

			if positive:
				pos += 1
			else:
				neg += 1

			for j in reversed(range(0, len(episode))):
				(X, y, action, reward, terminal) = episode[j]
				sample = [X, self.model.predict(X)]

				if reward != 0:
					mod = 1. if positive else -1.
				else:
					mod *= .99

				sample[1][action] = mod
				X_t.append(sample[0])
				y_t.append(sample[1])

		return np.asarray(X_t), np.asarray(y_t)
