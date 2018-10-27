from __future__ import print_function
import numpy as np

from keras.optimizers import SGD
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
#from SpeechResearch import loadData

from sklearn.preprocessing import LabelEncoder
import pandas

'exception_verbosity = high'
batch_size = 5
hidden_units = 13
nb_classes = 10
print('Loading data...')


# load train dataset
dataframe = pandas.read_csv("train.csv", header=None)
dataset = dataframe.values
X_train = dataset[:,0:13] #.astype(float)
Y = dataset[:,13]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
# y_train = np_utils.to_categorical(encoded_Y)


# load test dataset
dataframe = pandas.read_csv("test.csv", header=None)
dataset = dataframe.values
X_test = dataset[:,0:13] #.astype(float)
y_test = dataset[:,13]
# encode class values as integers
encoder2 = LabelEncoder()
encoder2.fit(y_test)
encoded_Y2 = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
# y_test = np_utils.to_categorical(encoded_Y2)



#(X_train, y_train), (X_test, y_test) = loadData.load_mfcc(10, 2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# print('y_train shape:', y_train.shape)
# print('y_test shape:', y_test.shape)
# print(y_test)
print('Build model...')

X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(encoded_Y, nb_classes)
Y_test = np_utils.to_categorical(encoded_Y2, nb_classes)
print(Y_train.shape)
print(Y_test.shape)
Y_train = Y_train.reshape(1, Y_train.shape[0], Y_train.shape[1])
Y_test = Y_test.reshape(1, Y_test.shape[0], Y_test.shape[1])

model = Sequential()
model.add(LSTM(units=hidden_units, kernel_initializer='uniform',
           unit_forget_bias='one', activation='tanh', recurrent_activation='sigmoid', input_shape=(None,X_train.shape[2]),     return_sequences=True))


model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, epochs=3)
score, acc = model.evaluate(X_test, Y_test,
                        batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
