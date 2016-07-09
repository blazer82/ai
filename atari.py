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
		inputs = np.zeros((min(len_memory, batch_size), 4, 210, 160))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			input_t, action_t, reward_t, input_tp1 = self.memory[idx][0]
			game_over = self.memory[idx][1]

			inputs[i:i+1] = input_t

			targets[i] = model.predict(input_t.reshape(1, 4, 210, 160))[0]
			q_next = np.max(model.predict(input_tp1.reshape(1, 4, 210, 160))[0])

			if game_over:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * q_next

		return inputs, targets

def make_grey(x):
	return np.average(x, 2)

if __name__ == "__main__":
	episodes = 1000
	epsilon = 1. # exploration
	epsilon_degrade = .000001
	epsilon_min = .1
	skip_frames = 4

	model = Sequential()
	model.add(Convolution2D(32, 8, 8,
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same',
		input_shape=(4, 210, 160),
		activation='relu'))
	model.add(Convolution2D(64, 4, 4,
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same',
		activation='relu'))
	model.add(Convolution2D(64, 3, 3,
		subsample=(1, 1),
		dim_ordering='th',
		border_mode='same',
		activation='relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dense(6))
	model.compile(sgd(lr=.2), "mse")

	exp_replay = ExperienceReplay(max_memory=500)

	env = gym.make('Breakout-v0')

	win_count = 0

	for i_episode in range(episodes):
		loss = 0.
		frame = 0
		game_over = False
		input = np.zeros((4, 210, 160))
		observation = env.reset() # shape (210, 160, 3)
		action = env.action_space.sample()

		while not game_over:
			env.render()

			input_tm1 = input

			if frame > 3:
				frame_iteration = 3
				input[0] = input[1]
				input[1] = input[2]
				input[2] = input[3]
			else:
				frame_iteration = frame%skip_frames

			if frame_iteration != 3:
				observation, reward, game_over, info = env.step(action)
				input[frame_iteration] = make_grey(observation)
				win_count += reward
			else:
				if np.random.rand() <= epsilon:
					action = env.action_space.sample()
				else:
					q = model.predict(input.reshape(1, 4, 210, 160))
					action = np.argmax(q)

				observation, reward, game_over, info = env.step(action)
				input[frame_iteration] = make_grey(observation)
				win_count += reward

				exp_replay.remember([input_tm1, action, reward, input], game_over)

				inputs, targets = exp_replay.get_batch(model, batch_size=50)

				loss += model.train_on_batch(inputs, targets)

				if epsilon > epsilon_min:
					epsilon -= epsilon * epsilon_degrade

			frame += 1

		print "Episode %d, loss %f, win average %f"%(i_episode, loss, win_count / (i_episode + 1.))
