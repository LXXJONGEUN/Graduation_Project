import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from numpy import argmax

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255.0
y_test = np_utils.to_categorical(y_test, 10)


xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

model = load_model('test.h5')
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))