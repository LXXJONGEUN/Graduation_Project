import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from numpy import argmax
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255.0
y_test = np_utils.to_categorical(y_test, 10)


xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

model = load_model('./models/test.h5')
yhat = model.predict_classes(xhat)

images = xhat
images *= 255
images = images.astype('uint8')
images = images.reshape(5, 28, 28)

for i in range(5):
    img = Image.fromarray(images[i], 'L')
    img.save('./public/images/test_image_' + str(i) + '.png')
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))