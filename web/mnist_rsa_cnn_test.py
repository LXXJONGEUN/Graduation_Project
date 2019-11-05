import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from numpy import argmax
from PIL import Image
import crypto.RSA2 as rsa

# RSA setting#########################
p = 13
q = 23
n = p * q
totient = (p-1)*(q-1)
e = rsa.get_public_key(totient)
d = rsa.get_private_key(e, totient)
######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_test = np_utils.to_categorical(y_test, 10)

xhat_idx = np.random.choice(x_test.shape[0], 10)
xhat = x_test[xhat_idx]

for i in range(10):
    img = Image.fromarray(xhat[i], 'L')
    img.save('./public/images/test_image_' + str(i) + '.png')

projections_test = []
for step in range(len(xhat)):
    test_image = xhat[step]
    projection_x = [0 for z in range(28)]
    projection_y = [0 for z in range(28)]
    for i in range(28):
        for j in range(28):
            if test_image[i][j] > 128:
                test_image[i][j] = 1
                projection_x[i] += 1
                projection_y[j] += 1
            else:
                test_image[i][j] = 0

    for i in range(28):
        if projection_x[i] < 10:
            temp = rsa.encrypt((e,n), "0" + str(projection_x[i]))
        else:
            temp = rsa.encrypt((e,n), str(projection_x[i]))
        projection_x[i] = temp[0] + temp[1]
    for i in range(28):
        if projection_y[i] < 10:
            temp = rsa.encrypt((e,n), "0" + str(projection_y[i]))
        else:
            temp = rsa.encrypt((e,n), str(projection_y[i]))
        projection_y[i] = temp[0] + temp[1]

    projection_x = [int(val) for val in projection_x]
    projection_y = [int(val) for val in projection_y]
    encrypt_projection = []
    encrypt_projection.append(projection_x)
    encrypt_projection.append(projection_y)

    projections_test.append(encrypt_projection)

projections_test = np.array(projections_test)
projections_test = projections_test.astype('float32')
projections_test = projections_test[:, :, :, np.newaxis]

model = load_model('./models/mnist_rsa_cnn.h5')
yhat = model.predict_classes(projections_test)

for i in range(10):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))