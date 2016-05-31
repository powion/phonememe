import sys
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

phoneme_map = np.load('phoneme_map.npy').tolist()
num_phonemes = len(phoneme_map)

Y_pred = np.memmap('Y_pred.mat', dtype='float32', mode='r')
num_samples = Y_pred.shape[0]//num_phonemes
Y_pred.shape = (num_samples, num_phonemes)
y_pred = Y_pred.argmax(axis=1)
y_test = np.memmap('y_test.mat', dtype='float32', mode='r')

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
plt.tight_layout()
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('confmat.pdf')
