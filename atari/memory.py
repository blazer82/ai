import numpy as np


class Memory:
	def __init__(self, model, size=100, episode_max_size=100):
		self.size = size
		self.episode_max_size = episode_max_size
		self.model = model
		self.memory = []

	def add(self, episode, positive):
		if len(self.memory) >= self.size:
			self.memory = self.memory[1:]

		episode = np.asarray(episode)
		if not self.episode_max_size is None:
			episode = episode[-self.episode_max_size:]
		self.memory.append([episode, positive])

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

			if positive is True and pos >= nbr_positive: continue
			if positive is False and neg >= nbr_negative: continue

			if positive is True:
				pos += 1
			else:
				neg += 1

			for j in reversed(range(0, len(episode))):
				(X, y, action, reward, terminal) = episode[j]
				sample = [X, np.zeros_like(y)]

				if reward != 0:
					mod = 1. if positive else -1.
				else:
					mod *= .99

				sample[1] = y * mod
				X_t.append(sample[0])
				y_t.append(sample[1])

		return np.asarray(X_t), np.asarray(y_t)
