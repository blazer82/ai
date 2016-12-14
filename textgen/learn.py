# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing import sequence
# load ascii text and covert to lowercase
filename = "alice.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = re.sub('\s+', ' ', raw_text)
raw_text = re.sub('[^a-z.!? ]', '', raw_text)
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_len = 140
sentences = re.split('[.!?]\s+', raw_text)
sentences = filter(lambda s: len(s) > 10, sentences)
n_sentences = len(sentences)
print "Total Sentences: ", n_sentences
dataX = []
dataY = []
for s in range(0, n_sentences):
	sentence = sentences[s]
	for i in range(0, len(sentence) - 1, 1):
		seq_in = sentence[0:i]
		seq_out = sentence[i]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
dataX = sequence.pad_sequences(dataX, maxlen=seq_len)
X = numpy.reshape(dataX, (n_patterns, len(dataX[1]), 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(Convolution1D(32, 3, input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=20, batch_size=128, callbacks=callbacks_list, verbose=1)
