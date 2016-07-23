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
	epsilon_degrade = 0
	epsilon_min = .1
	discount = .99
	episodes_per_batch = 10
	batch_size = 128

	model = Sequential()

	model.add(Convolution2D(32, 8, 8,
		init='glorot_uniform',
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same',
		input_shape=(2, 80, 74)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 4, 4,
		init='glorot_uniform',
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3,
		init='glorot_uniform',
		subsample=(1, 1),
		dim_ordering='th',
		border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(512, init='glorot_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Dense(6, init='glorot_uniform'))
	model.add(Activation('softmax'))

	model.compile(RMSprop(lr=25e-5), loss='mse')

	env = gym.make('Pong-v0')

	observation = env.reset()
	episode = 0
	score = 0
	prev_x = np.zeros((80, 74))
	xs = []
	qs = []
	ys = []
	actions = []
	rewards = []
	losses = []
	scores = []
	mean_scores = []
	max_scores = []
	meanqs = []
	e = epsilon

	while True:
		#env.render()

		# preprocess observation and calculate frame difference
		cur_x = preprocess(observation)
		x = np.asarray([prev_x, cur_x])
		prev_x = cur_x

		xs.append(x)

		# determine next action
		q = model.predict(x.reshape(1, 2, 80, 74))[0]
		action = np.argmax(q) if np.random.rand() > e else np.random.choice(6, 1, p=q/np.sum(q))[0]

		y = q.copy()
		y[action] = 1. # Encourage current action

		qs.append(q)
		actions.append(action)
		ys.append(y)

		# execute step and make observation
		observation, reward, terminal, info = env.step(action)

		rewards.append(reward)
		score += reward

		if terminal:
			episode += 1
			prev_x = np.zeros((80, 74))
			scores.append(score)
			score = 0
			observation = env.reset()

			if episode%episodes_per_batch == 0:
				exs = np.asarray(xs)
				eqs = np.asarray(qs)
				eys = np.asarray(ys)
				erewards = np.asarray(rewards)

				# discount rewards
				dr = np.zeros(erewards.shape)
				ra = 0
				for i in reversed(range(0, erewards.size)):
					if erewards[i] != 0: ra = 0 # Pong specific?
					ra = ra * discount + erewards[i]
					dr[i] = ra
					#print "discount %d:%f -> %f"%(i, erewards[i], dr[i])
				dr -= np.mean(dr)
				dr /= np.std(dr) if np.std(dr) > 0. else 1.

				for i,a in enumerate(actions):
					eys[i][a] *= dr[i]

				if batch_size == None:
					loss = model.train_on_batch(exs, eys)
				else:
					history = model.fit(exs, eys, nb_epoch=1, batch_size=batch_size, verbose=1)
					loss = history.history['loss'][0]

				meanq = np.mean(np.max(qs, axis=1))
				mean_score = np.mean(scores)
				max_score = np.max(scores)
				losses.append(loss)
				mean_scores.append(mean_score)
				max_scores.append(max_score)
				meanqs.append(meanq)

				# log
				print "Episode %d, loss %f, mean q %f, epsilon %f, mean score %f"%(episode, loss, meanq, e, mean_score)

				# plot
				if episode > episodes_per_batch:
					plt.close()
					s_loss = plt.subplot(311)
					s_q = plt.subplot(312)
					s_score = plt.subplot(313)
					s_loss.plot(range(0, len(losses)), losses, 'b-')
					s_q.plot(range(0, len(meanqs)), meanqs, 'b-')
					s_score.plot(range(0, len(mean_scores)), mean_scores, 'b-')
					if episodes_per_batch > 1:
						s_score.plot(range(0, len(max_scores)), max_scores, 'g-')
					plt.show(block=False)

				# reset
				xs = []
				qs = []
				ys = []
				actions = []
				rewards = []
				scores = []

				if e > epsilon_min:
					e = epsilon - epsilon_degrade*episode

				# save model and plot
				print "Saving weights to disk..."
				model.save_weights('pg-atari_weights.h5', overwrite=True)
				print "Saving plot to disk..."
				plt.savefig('pg-atari_plot.png')
