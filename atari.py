import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class ExperienceReplay(object):
	def __init__(self, max_memory=100, discount=.9):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount

	def remember(self, states, game_over):
		self.memory.append([states, game_over])
		if len(self.memory) > self.max_memory:
			del self.memory[0]

	def get_batch(self, model, batch_size):
		len_memory = len(self.memory)
		num_actions = 6
		env_dim = 100800 #self.memory[0][0][0].shape[1]
		inputs = np.zeros((min(len_memory, batch_size), env_dim))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			input_t, action_t, reward_t, input_tp1 = self.memory[idx][0]
			game_over = self.memory[idx][1]

			inputs[i:i+1] = input_t

			targets[i] = model.predict(input_t)[0]
			q_next = np.max(model.predict(input_tp1)[0])

			if game_over:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * q_next

		return inputs, targets

if __name__ == "__main__":
	episodes = 1000
	epsilon = 1. # exploration
	epsilon_degrade = .0001
	epsilon_min = .2

	model = Sequential()
	model.add(Dense(100, input_shape=(100800,), activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(6))
	model.compile(sgd(lr=.2), "mse")

	exp_replay = ExperienceReplay(max_memory=500)

	env = gym.make('Breakout-v0')

	win_count = 0

	for i_episode in range(episodes):
		loss = 0.
		observation = env.reset() # shape (210, 160, 3)
		input = np.array(observation, dtype=float).flatten().reshape((1, -1))
		game_over = False

		while not game_over:
			observation_tm1 = observation
			input_tm1 = input
			env.render()

			if np.random.rand() <= epsilon:
				action = env.action_space.sample()
			else:
				input = np.array(observation, dtype=float).flatten().reshape((1, -1))
				q = model.predict(input)
				action = np.argmax(q)

			observation, reward, game_over, info = env.step(action)

			if reward == 1:
				win_count += 1

			exp_replay.remember([input_tm1, action, reward, input], game_over)

			inputs, targets = exp_replay.get_batch(model, batch_size=20)

			loss += model.train_on_batch(inputs, targets)

			if epsilon > epsilon_min:
				epsilon -= epsilon * epsilon_degrade

		print "Episode %d, loss %f, win count %d"%(i_episode, loss, win_count)
