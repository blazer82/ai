import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from PIL import Image


def preprocess(x):
	grey = np.average(x, 2)
	img = Image.fromarray(grey)
	img.thumbnail((110, 110), Image.NEAREST) # new shape (110, 84)
	img = img.crop((5, 25, 79, 105)) # new shape (80, 74)
	x_new = np.asarray(img, dtype=np.float32).copy()
	x_new -= np.mean(x_new)
	return x_new

if __name__ == "__main__":
	epsilon = 1. # exploration
	epsilon_degrade = 1e-4
	epsilon_min = .1
	discount = .99

	model = Sequential()

	model.add(Convolution2D(16, 8, 8,
		init='glorot_uniform',
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same',
		input_shape=(2, 80, 74)))
	model.add(Activation('relu'))

	model.add(Convolution2D(32, 4, 4,
		init='glorot_uniform',
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same'))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(256, init='glorot_uniform'))
	model.add(Activation('relu'))

	model.add(Dense(6, init='glorot_uniform'))
	model.add(Activation('softmax'))

	model.compile(RMSprop(lr=1e-5), loss='mse')

	env = gym.make('Pong-v0')

	observation = env.reset()
	episode = 0
	score = 0
	prev_x = np.zeros((80, 74))
	xs = []
	qs = []
	actions = []
	rewards = []
	losses = []
	scores = []
	meanqs = []
	e = epsilon

	while True:
		env.render()

		# preprocess observation and calculate frame difference
		cur_x = preprocess(observation)
		x = np.asarray([prev_x, cur_x])
		prev_x = cur_x

		xs.append(x)

		# determine next action
		q = model.predict(x.reshape(1, 2, 80, 74))[0]
		action = np.argmax(q) if np.random.rand() > e else np.random.randint(0, 6)

		qs.append(q)
		actions.append(action)

		# execute step and make observation
		observation, reward, terminal, info = env.step(action)

		rewards.append(reward)
		score += reward

		if terminal:
			episode += 1

			exs = np.asarray(xs)
			eqs = np.asarray(qs)
			erewards = np.asarray(rewards)

			# discount rewards
			dr = np.zeros(erewards.shape)
			ra = 0
			for i in reversed(range(0, erewards.size)):
				ra = ra * discount + erewards[i]
				dr[i] = ra
			dr -= np.mean(dr)

			for i,q in enumerate(eqs):
				eqs[i][actions[i]] += dr[i]

			targets = np.asarray(eqs)

			bx = exs.reshape(exs.shape[0], 2, 80, 74)
			loss = model.train_on_batch(bx, targets)


			meanq = np.mean(np.max(qs, axis=1))
			losses.append(loss)
			scores.append(score)
			meanqs.append(meanq)

			# log
			print "Episode %d, loss %f, mean q %f, epsilon %f, score %d"%(episode, loss, meanq, e, score)

			# plot
			if episode > 3:
				plt.close()
				s_loss = plt.subplot(311)
				s_q = plt.subplot(312)
				s_score = plt.subplot(313)
				s_loss.plot(range(2, episode), losses[2:], 'b-')
				s_q.plot(range(0, episode), meanqs, 'b-')
				s_score.plot(range(0, episode), scores, 'b-')
				plt.show(block=False)

			# reset
			xs = []
			qs = []
			actions = []
			rewards = []
			prev_x = np.zeros((80, 74))
			score = 0
			observation = env.reset()

			if e > epsilon_min:
				e = epsilon - epsilon_degrade*episode

			# save model and plot
			if episode%10 == 0:
				print "Saving weights to disk..."
				model.save_weights('pg-atari_weights.h5', overwrite=True)
				print "Saving plot to disk..."
				plt.savefig('pg-atari_plot.png')
