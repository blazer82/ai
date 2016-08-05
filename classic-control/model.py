from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


class Model:
	def __init__(self, init='normal', activation='relu', batch_size=32, lr=1e-2, load=None):
		self.batch_size = batch_size
		self.model = Sequential()

		self.model.add(Convolution1D(8, 1, input_shape=(2, 4), init=init))
		self.model.add(Activation(activation))

		self.model.add(Flatten())

		self.model.add(Dense(8, init=init))
		self.model.add(BatchNormalization())
		self.model.add(Activation(activation))

		self.model.add(Dense(8, init=init))
		self.model.add(BatchNormalization())
		self.model.add(Activation(activation))

		self.model.add(Dense(2, init=init))

		if load != None:
			self.model.load_weights(load)

		self.model.compile(SGD(lr=lr), loss='mse')

	def predict(self, X):
		return self.model.predict(X.reshape((1,) + X.shape))[0]

	def learn(self, X, y):
		return self.model.fit(X, y, nb_epoch=1, batch_size=self.batch_size, shuffle=True, verbose=0)

	def save(self, filename):
		return self.model.save_weights(filename, overwrite=True)
