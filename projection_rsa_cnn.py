# INFO ########################################
# 암호화 데이터 타입 : 이미지(MNIST)
# 암호화 알고리즘 : RSA
# 암호화 적용 방식 : projection 후 암호화
# 학습 알고리즘 : CNN
# 현재 정확도 78%(epoch: 20)
# comment : epoch을 더 늘리면 정확도 올라갈 듯
# Test accuracy:  0.74856
###############################################

import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import crypto.RSA2 as rsa

# RSA setting#########################
p = 13
q = 23
n = p * q
totient = (p-1)*(q-1)
e = rsa.get_public_key(totient)
d = rsa.get_private_key(e, totient)
######################################

(X_train, y_train), (X_test, y_test) = mnist.load_data()


projections_train = []
projections_not_encrypt_train = []
for step in range(len(X_train)):
    train_image = X_train[step]
    projection = []
    projection_x = [0 for z in range(28)]
    projection_y = [0 for z in range(28)]
    for i in range(28):
        for j in range(28):
            if train_image[i][j] > 128:
                train_image[i][j] = 1
                projection_x[i] += 1
                projection_y[j] += 1
            else:
                train_image[i][j] = 0

    projection.append(projection_x)
    projection.append(projection_y)
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

    projections_train.append(encrypt_projection)
    projections_not_encrypt_train.append(projection)


projections_test = []
projections_not_encrypt_test = []
for step in range(len(X_test)):
    test_image = X_test[step]
    projection = []
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

    projection.append(projection_x)
    projection.append(projection_y)

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
    projections_not_encrypt_test.append(projection)


projections_train = np.array(projections_train)
projections_test = np.array(projections_test)
projections_not_encrypt_train = np.array(projections_not_encrypt_train)
projections_not_encrypt_test = np.array(projections_not_encrypt_test)

projections_train = projections_train.astype('float32')
projections_test = projections_test.astype('float32')
projections_not_encrypt_train = projections_not_encrypt_train.astype('float32')
projections_not_encrypt_test = projections_not_encrypt_test.astype('float32')

projections_train = projections_train[:, :, :, np.newaxis]
projections_test = projections_test[:, :, :, np.newaxis]
projections_not_encrypt_train = projections_not_encrypt_train[:, :, :, np.newaxis]
projections_not_encrypt_test = projections_not_encrypt_test[:, :, :, np.newaxis]
print(type(projections_train))
print(projections_train.shape)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# projections_train = projections_train.reshape(-1, 56, 1)
# projections_test = projections_test.reshape(-1 ,56, 1)
# print(projections_train.shape)
# print(projections_test.shape)
################################################
print("start")
model = Sequential()
model_unencrypt = Sequential()
model.add(Conv2D(32, kernel_size=2, padding='same', input_shape=(2, 28, 1)))
model_unencrypt.add(Conv2D(32, kernel_size=2, padding='same', input_shape=(2, 28, 1)))
model.add(Activation('relu'))
model_unencrypt.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model_unencrypt.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=2, padding='same'))
model_unencrypt.add(Conv2D(64, kernel_size=2, padding='same'))
model.add(Activation('relu'))
model_unencrypt.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model_unencrypt.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=2, padding='same'))
model_unencrypt.add(Conv2D(128, kernel_size=2, padding='same'))
model.add(Activation('relu'))
model_unencrypt.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model_unencrypt.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=2, padding='same'))
model_unencrypt.add(Conv2D(256, kernel_size=2, padding='same'))
model.add(Activation('relu'))
model_unencrypt.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model_unencrypt.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=2, padding='same'))
model_unencrypt.add(Conv2D(256, kernel_size=2, padding='same'))
model.add(Activation('relu'))
model_unencrypt.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model_unencrypt.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model_unencrypt.add(Dropout(0.3))

model.add(Flatten())
model_unencrypt.add(Flatten())
model.add(Dense(10))
model_unencrypt.add(Dense(10))
model.add(Activation('softmax'))
model_unencrypt.add(Activation('softmax'))
model.summary()
model_unencrypt.summary()

model.compile(loss='categorical_crossentropy',
                optimizer=Adam(), metrics=['accuracy'])
model_unencrypt.compile(loss='categorical_crossentropy',
                optimizer=Adam(), metrics=['accuracy'])

history_encrypt = model.fit(projections_train, y_train,
                    batch_size=128, epochs=20,
                    verbose=1, validation_split=0.2)

history_not_encrypt = model_unencrypt.fit(projections_not_encrypt_train, y_train,
                    batch_size=128, epochs=20,
                    verbose=1, validation_split=0.2)

score = model.evaluate(projections_test, y_test, verbose=1)
score2 = model_unencrypt.evaluate(projections_not_encrypt_test, y_test, verbose=1)
print("\nEncrypt Test score: ", score[0])
print("\nEncrypt Test accuracy: ", score[1])
print("\n\nNot Encrypt Test score: ", score2[0])
print("\nNot Encrypt Test accuracy: ", score2[1])

from keras.models import load_model
model.save('mnist_rsa_cnn.h5')


import matplotlib.pyplot as plt

acc = history_not_encrypt.history['acc']
loss = history_not_encrypt.history['loss']

encrypt_acc = history_encrypt.history['acc']
encrypt_loss = history_encrypt.history['loss']

epochs = range(1, len(acc)+1)

plt.clf()

plt.plot(epochs, loss, 'r', label='Loss (Before encryption)')
plt.plot(epochs, encrypt_loss, 'b', label='Loss (After encryption)')
plt.title('Loss (Before & After)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

plt.plot(epochs, acc, 'r', label='Accuracy (Before encryption)')
plt.plot(epochs, encrypt_acc, 'b', label='Accuracy (After encryption)')
plt.title('Accuracy (Before & After)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()