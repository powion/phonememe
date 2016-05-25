'''Trains a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Masking
from keras.regularizers import l2, activity_l2

batch_size, num_batches = 4096, 20
length, num_features = 20, 40

print('Loading data...')

phoneme_map = np.load('phoneme_map.npy').tolist()
num_phonemes = len(phoneme_map)

X_train = np.memmap('X_train.mat', dtype='float32', mode='r')
y_train = np.memmap('y_train.mat', dtype='float32', mode='r')
Y_train = np_utils.to_categorical(y_train, num_phonemes)

X_test = np.memmap('X_test.mat', dtype='float32', mode='r')
y_test = np.memmap('y_test.mat', dtype='float32', mode='r')
Y_test = np_utils.to_categorical(y_test, num_phonemes)

num_train_samples = X_train.shape[0]//length//num_features
num_test_samples = X_test.shape[0]//length//num_features
X_train.shape = (num_train_samples, length, num_features)
X_test.shape = (num_test_samples, length, num_features)

print('X_train shape:', X_train.shape)
print('X_test shape: ', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(length, num_features)))
model.add(LSTM(128, input_shape=(length, num_features)))
model.add(Dense(num_phonemes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop')

print('Train...')
model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=num_batches,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            show_accuracy=True,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
