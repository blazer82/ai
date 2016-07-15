import gym
import numpy as np
from scipy import misc
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import sgd
from PIL import Image


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
		encouraged_actions = np.zeros(num_actions, dtype=np.int)
		predicted_actions = np.zeros(num_actions, dtype=np.int)
		inputs = np.zeros((min(len_memory, batch_size), 4, 80, 74))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			input_t, action_t, reward_t, input_tp1 = self.memory[idx][0]
			game_over = self.memory[idx][1]

			inputs[i] = input_t

			targets[i] = model.predict(input_t.reshape(1, 4, 80, 74))[0]
			q_next = np.max(model.predict(input_tp1.reshape(1, 4, 80, 74))[0])

			predicted_actions[np.argmax(targets[i])] += 1

			if game_over:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * q_next

			encouraged_actions[np.argmax(targets[i])] += 1

		return inputs, targets, encouraged_actions, predicted_actions

def preprocess(x):
	grey = np.average(x, 2)
	img = Image.fromarray(grey)
	img.thumbnail((110, 110), Image.NEAREST) # new shape (110, 84)
	img = img.crop((5, 25, 79, 105)) # new shape (80, 74)
	x_new = np.asarray(img, dtype=np.float32)
	return x_new

if __name__ == "__main__":
	episodes = 100000
	epsilon = 1. # exploration
	epsilon_degrade = .000001
	epsilon_min = .1
	skip_frames = 4

	model = Sequential()
	model.add(Convolution2D(16, 8, 8,
		init='uniform',
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same',
		input_shape=(4, 80, 74)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 4, 4,
		init='uniform',
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same'))
	model.add(Activation('relu'))

	"""model.add(Convolution2D(64, 3, 3,
		init='uniform',
		subsample=(1, 1),
		dim_ordering='th',
		border_mode='same'))
	model.add(Activation('relu'))"""

	model.add(Flatten())

	model.add(Dense(256, init='uniform'))
	model.add(Activation('relu'))

	model.add(Dense(6, init='uniform'))
	model.add(Activation('softmax'))

	model.compile(sgd(lr=.1), "mse")

	exp_replay = ExperienceReplay(max_memory=100000)

	env = gym.make('Breakout-v0')

	total_score = 0
	total_frames = 0

	for i_episode in range(episodes):
		loss = 0.
		frame = 0
		frame_index = 0
		score = 0
		game_over = False
		input = np.zeros((4, 80, 74))
		observation = env.reset() # shape (210, 160, 3)
		action = env.action_space.sample()
		encouraged_actions = np.zeros(6, dtype=np.int)
		predicted_actions = np.zeros(6, dtype=np.int)

		while not game_over:
			env.render()

			input_tm1 = input

			if frame%skip_frames != 3:
				observation, reward, game_over, info = env.step(action)
				score += reward
			else:
				if np.random.rand() <= epsilon:
					action = env.action_space.sample()
				else:
					q = model.predict(input.reshape(1, 4, 80, 74))
					action = np.argmax(q)

				observation, reward, game_over, info = env.step(action)

			if frame%skip_frames == 3 or game_over:
				if frame_index == 4:
					frame_index = 3
					input[0:2] = input[1:3]
					input[3] = np.zeros(input[3].shape)

				input[frame_index] = preprocess(observation)
				score += reward

				if game_over:
					reward = -1

				exp_replay.remember([input_tm1, action, reward, input], game_over)

				inputs, targets, encouraged, predicted = exp_replay.get_batch(model, batch_size=32)

				encouraged_actions += encouraged
				predicted_actions += predicted

				loss += model.train_on_batch(inputs, targets)

				frame_index += 1

			if epsilon > epsilon_min:
				epsilon -= epsilon * epsilon_degrade
				epsilon = max(epsilon, epsilon_min)

			frame += 1

		total_frames += frame
		total_score += score
		print "Episode %d, loss %f, score %d"%(i_episode, loss, score)
		print "Predicted actions 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d"%(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3], predicted_actions[4], predicted_actions[5])
		print "Encouraged actions 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d"%(encouraged_actions[0], encouraged_actions[1], encouraged_actions[2], encouraged_actions[3], encouraged_actions[4], encouraged_actions[5])
		print "Frames %d, epsilon %f"%(total_frames, epsilon)

		if i_episode > 0 and i_episode%50 == 0:
			print "Saving weights to disk..."
			model.save_weights('atari_weights.h5', overwrite=True)
