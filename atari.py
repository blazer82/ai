import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
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
		inputs = np.zeros((min(len_memory, batch_size), 210, 160, 3))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			input_t, action_t, reward_t, input_tp1 = self.memory[idx][0]
			game_over = self.memory[idx][1]

			inputs[i:i+1] = input_t

			targets[i] = model.predict(input_t.reshape(1, 210, 160, 3))[0]
			q_next = np.max(model.predict(input_tp1.reshape(1, 210, 160, 3))[0])

			if game_over:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * q_next

		return inputs, targets

if __name__ == "__main__":
	episodes = 1000
	epsilon = 1. # exploration
	epsilon_degrade = .0001
	epsilon_min = .1
	skip_frames = 4

	model = Sequential()
	model.add(Convolution2D(32, 8, 8,
		subsample=(4, 4),
		dim_ordering='tf',
		border_mode='same',
		input_shape=(210, 160, 3),
		activation='relu'))
	model.add(Convolution2D(64, 4, 4,
		subsample=(2, 2),
		dim_ordering='tf',
		border_mode='same',
		activation='relu'))
	model.add(Convolution2D(64, 3, 3,
		subsample=(1, 1),
		dim_ordering='tf',
		border_mode='same',
		activation='relu'))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Dense(6, activation='softmax'))
	model.compile(sgd(lr=.2), "mse")

	exp_replay = ExperienceReplay(max_memory=500)

	env = gym.make('Breakout-v0')

	win_count = 0

	for i_episode in range(episodes):
		loss = 0.
		frame = 0
		observation = env.reset() # shape (210, 160, 3)
		input = np.array(observation, dtype=float)
		game_over = False
		action = env.action_space.sample()

		while not game_over:
			env.render()

			if frame%skip_frames != 0:
				env.step(action)
			else:
				observation_tm1 = observation
				input_tm1 = input

				if np.random.rand() <= epsilon:
					action = env.action_space.sample()
				else:
					input = np.array(observation, dtype=float)
					q = model.predict(input.reshape(1, 210, 160, 3))
					action = np.argmax(q)

				observation, reward, game_over, info = env.step(action)

				if reward == 1:
					win_count += 1

				exp_replay.remember([input_tm1, action, reward, input], game_over)

				inputs, targets = exp_replay.get_batch(model, batch_size=20)

				loss += model.train_on_batch(inputs, targets)

				if epsilon > epsilon_min:
					epsilon -= epsilon * epsilon_degrade

			frame += 1

		print "Episode %d, loss %f, win average %f"%(i_episode, loss, win_count / (i_episode + 1))
