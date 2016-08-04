from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop


class Model:
	def __init__(self, init='glorot_uniform', activation='relu', batch_size=32, load=None):
		self.batch_size = batch_size
		self.model = Sequential()

		self.model.add(Convolution2D(16, 8, 8,
			init=init,
			subsample=(4, 4),
			dim_ordering='th',
			border_mode='valid',
			input_shape=(2, 80, 74)))
		self.model.add(Activation(activation))

		self.model.add(Convolution2D(32, 4, 4,
			init=init,
			subsample=(2, 2),
			dim_ordering='th',
			border_mode='valid'))
		self.model.add(Activation(activation))

		self.model.add(Convolution2D(32, 3, 3,
			init=init,
			subsample=(1, 1),
			dim_ordering='th',
			border_mode='valid'))
		self.model.add(Activation(activation))

		self.model.add(Flatten())

		self.model.add(Dense(256, init=init))
		self.model.add(Activation(activation))

		self.model.add(Dense(6, init=init))
		self.model.add(Activation('softmax'))

		if load != None:
			self.model.load_weights(load)

		self.model.compile(RMSprop(lr=25e-5), loss='mse')

	def predict(self, X):
		return self.model.predict(X.reshape((1,) + X.shape))[0]

	def learn(self, X, y):
		return self.model.fit(X, y, nb_epoch=1, batch_size=self.batch_size, shuffle=True, verbose=1)

	def save(self, filename):
		return self.model.save_weights(filename, overwrite=True)
