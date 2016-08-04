import gym
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adam, rmsprop
from PIL import Image


space = {
	'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
	'nb_epochs':  20,
	'optimizer': hp.choice('optimizer', ['sgd', 'adadelta', 'adam', 'rmsprop']),
	'nb_layers': hp.choice('nb_layers', [4, 5]),
	'batch_normalization': hp.choice('batch_normalization', [False, True]),
	'activation': 'relu',
	'init': 'glorot_uniform'
}


def preprocess(x):
	grey = np.average(x, 2)
	img = Image.fromarray(grey)
	img.thumbnail((110, 110), Image.NEAREST) # new shape (110, 84)
	img = img.crop((5, 25, 79, 105)) # new shape (80, 74)
	x_new = np.asarray(img, dtype=np.float32).copy()
	x_new -= np.mean(x_new)
	return x_new


def sample_game(episodes = 10):
	env = gym.make('Pong-v0')

	prev_x = np.zeros((80, 74))
	xs = []
	ys = []
	actions = []
	rewards = []
	episode = 0
	discount = .99

	observation = env.reset()

	while True:
		cur_x = preprocess(observation)
		x = np.asarray([prev_x, cur_x])
		prev_x = cur_x

		xs.append(x)

		action = np.random.randint(0, 6)

		y = np.zeros(6)
		y[action] = 1.

		actions.append(action)
		ys.append(y)

		observation, reward, terminal, info = env.step(action)

		rewards.append(reward)

		if terminal:
			episode += 1
			prev_x = np.zeros((80, 74))
			observation = env.reset()

			if episode == episodes:
				exs = np.asarray(xs)
				eys = np.asarray(ys)
				erewards = np.asarray(rewards)

				# discount rewards
				dr = np.zeros(erewards.shape)
				ra = 0
				for i in reversed(range(0, erewards.size)):
					if erewards[i] != 0: ra = 0 # Pong specific?
					ra = ra * discount + erewards[i]
					dr[i] = ra
				dr -= np.mean(dr)
				dr /= np.std(dr) if np.std(dr) > 0. else 1.

				for i,a in enumerate(actions):
					eys[i][a] *= dr[i]

				return exs, eys


x, y = sample_game()


def f_nn(params):
	print ('Params testing: ', params)

	model = Sequential()

	model.add(Convolution2D(16, 8, 8,
		init=params['init'],
		subsample=(4, 4),
		dim_ordering='th',
		border_mode='same',
		input_shape=(2, 80, 74)))
	if params['batch_normalization']:
		model.add(BatchNormalization())
	model.add(Activation(params['activation']))

	model.add(Convolution2D(32, 4, 4,
		init=params['init'],
		subsample=(2, 2),
		dim_ordering='th',
		border_mode='same'))
	if params['batch_normalization']:
		model.add(BatchNormalization())
	model.add(Activation(params['activation']))

	if params['nb_layers'] == 5:
		model.add(Convolution2D(32, 3, 3,
			init=params['init'],
			subsample=(1, 1),
			dim_ordering='th',
			border_mode='same'))
		if params['batch_normalization']:
			model.add(BatchNormalization())
		model.add(Activation(params['activation']))

	model.add(Flatten())

	model.add(Dense(256, init=params['init']))
	if params['batch_normalization']:
		model.add(BatchNormalization())
	model.add(Activation(params['activation']))

	model.add(Dense(6, init=params['init']))
	model.add(Activation('softmax'))

	model.compile(optimizer=params['optimizer'], loss='mse')

	history = model.fit(x, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], validation_split=.2, verbose = 1)

	mean_loss = np.mean(history.history['loss'])
	mean_vloss = np.mean(history.history['val_loss'])
	return {'loss': np.mean([mean_loss, mean_vloss]), 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
