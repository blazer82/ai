import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from PIL import Image


class ExperienceReplay(object):
	def __init__(self, max_memory=100, discount=.9):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount

	def remember(self, states, terminal):
		self.memory.append([states, terminal])
		if len(self.memory) > self.max_memory:
			del self.memory[0]

	def get_batch(self, model, batch_size):
		len_memory = len(self.memory)
		num_actions = 6
		encouraged_actions = np.zeros(num_actions, dtype=np.int)
		predicted_actions = np.zeros(num_actions, dtype=np.int)
		inputs = np.zeros((min(len_memory, batch_size), 4, 80, 74))
		targets = np.zeros((inputs.shape[0], num_actions))
		q_list = np.zeros(inputs.shape[0])
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			input_t, action_t, reward_t, input_tp1 = self.memory[idx][0]
			terminal = self.memory[idx][1]

			inputs[i] = input_t

			targets[i] = model.predict(input_t.reshape(1, 4, 80, 74))[0]
			q_next = np.max(model.predict(input_tp1.reshape(1, 4, 80, 74))[0])

			q_list[i] = np.max(targets[i])
			predicted_actions[np.argmax(targets[i])] += 1

			if terminal:
				targets[i, action_t] = reward_t
			else:
				targets[i, action_t] = reward_t + self.discount * q_next

			print "Action %d rewarded with %f"%(action_t, targets[i, action_t])

			encouraged_actions[np.argmax(targets[i])] += 1

		return inputs, targets, encouraged_actions, predicted_actions, np.average(q_list)

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
	epsilon_degrade = 1e-6
	epsilon_min = .1
	skip_frames = 4

	model = Sequential()

	model.add(BatchNormalization(input_shape=(4, 80, 74)))

	model.add(Convolution2D(16, 8, 8,
		init='uniform',
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 4, 4,
		init='uniform',
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same'))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(256, init='uniform'))
	model.add(Activation('relu'))

	model.add(Dense(6, init='uniform'))
	model.add(Activation('softmax'))

	model.compile(RMSprop(lr=1e-3), loss='mse')

	exp_replay = ExperienceReplay(max_memory=100000)

	env = gym.make('Breakout-v0')

	total_score = 0
	total_frames = 0

	plot_loss = list()
	plot_score = list()
	plot_q = list()

	for i_episode in range(episodes):
		loss = list()
		q_list = list()
		frame = 0
		frame_index = 0
		score = 0
		terminal = False
		input = np.zeros((4, 80, 74))
		observation = env.reset() # shape (210, 160, 3)
		action = env.action_space.sample()
		encouraged_actions = np.zeros(6, dtype=np.int)
		predicted_actions = np.zeros(6, dtype=np.int)

		while not terminal:
			env.render()

			input_tm1 = input

			if frame%skip_frames != 3:
				observation, reward, terminal, info = env.step(action)
				score += reward
			else:
				if np.random.rand() <= epsilon:
					action = np.random.randint(0, 6)
				else:
					q = model.predict(input.reshape(1, 4, 80, 74))[0]
					action = np.argmax(q)

				observation, reward, terminal, info = env.step(action)

			if frame%skip_frames == 3 or terminal:
				if frame_index == 4:
					frame_index = 3
					input[0:2] = input[1:3]
					input[3] = np.zeros(input[3].shape)

				input[frame_index] = preprocess(observation)
				score += reward

				if game_over:
					reward = -1

				exp_replay.remember([input_tm1, action, reward, input], terminal)

				inputs, targets, encouraged, predicted, q_avg = exp_replay.get_batch(model, batch_size=128)

				encouraged_actions += encouraged
				predicted_actions += predicted
				q_list.append(q_avg)

				loss.append(model.train_on_batch(inputs, targets))

				frame_index += 1

			if epsilon > epsilon_min:
				epsilon -= epsilon * epsilon_degrade
				epsilon = max(epsilon, epsilon_min)

			frame += 1

		total_frames += frame
		total_score += score
		loss_avg = np.average(loss)
		q_avg = np.average(q_list)
		print "Episode %d, mean loss %f, avg q %f, score %d"%(i_episode, loss_avg, q_avg, score)
		print "Predicted actions 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d"%(predicted_actions[0], predicted_actions[1], predicted_actions[2], predicted_actions[3], predicted_actions[4], predicted_actions[5])
		print "Encouraged actions 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d"%(encouraged_actions[0], encouraged_actions[1], encouraged_actions[2], encouraged_actions[3], encouraged_actions[4], encouraged_actions[5])
		print "Frames %d, epsilon %f"%(total_frames, epsilon)

		plot_loss.append(loss_avg)
		plot_q.append(q_avg)
		plot_score.append(score)
		plt.close()
		s_loss = plt.subplot(311)
		s_q = plt.subplot(312)
		s_score = plt.subplot(313)
		s_loss.plot(range(0, i_episode+1), plot_loss, 'b-')
		s_q.plot(range(0, i_episode+1), plot_q, 'b-')
		s_score.plot(range(0, i_episode+1), plot_score, 'b-')
		plt.show(block=False)

		if i_episode > 0 and i_episode%50 == 0:
			print "Saving weights to disk..."
			model.save_weights('atari_weights.h5', overwrite=True)
