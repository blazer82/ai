# Load LSTM network and generate text
import sys
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
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
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
model.add(Convolution1D(64, 3, input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = sys.argv[1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(140):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern = numpy.append(pattern, index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
