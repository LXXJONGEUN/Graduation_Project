# INFO ########################################
# 암호화 데이터 타입 : 이미지(MNIST)
# 암호화 알고리즘 : RSA
# 암호화 적용 방식 : projection 후 암호화
# 학습 알고리즘 : RNN
# 현재 정확도 : 51%
# comment : 층을 더 쌓거나, dropout하면 정확도 20~30%
#           underfitting으로 예상, 데이터 셋이 RNN과 맞지않는 듯
# Test accuracy:  0.5249
###############################################

import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import crypto.RSA2 as rsa
from keras import layers
from matplotlib import pyplot as plt

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

# projections_train = projections_train.reshape(60000, 56)
# projections_test = projections_test.reshape(10000, 56)

print(type(projections_train))
print(projections_train.shape)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

################################################
print("start")

model = Sequential()
model.add(layers.LSTM(64, input_shape=(2, 28)))
# model.add(layers.LSTM(32, return_sequences=True, input_shape=(2, 28)))
# # model.add(layers.LSTM(64, return_sequences=True, activation='relu'))
# # model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
# model.add(layers.LSTM(256, activation='relu'))

model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


model_unencrypt = Sequential()
model_unencrypt.add(layers.LSTM(64, input_shape=(2, 28)))

model_unencrypt.add(Dense(10))
model_unencrypt.add(Activation('softmax'))
model_unencrypt.summary()

model.compile(loss='mae', optimizer=RMSprop(), metrics=['accuracy'])
model_unencrypt.compile(loss='mae', optimizer=RMSprop(), metrics=['accuracy'])

epoch_num = 100

history_encrypt = model.fit(projections_train, y_train,
                    epochs=epoch_num, batch_size=128,
                    verbose=1, validation_split=0.2)

history_not_encrypt = model_unencrypt.fit(projections_not_encrypt_train, y_train,
                    epochs=epoch_num, batch_size=128,
                    verbose=1, validation_split=0.2)

score = model.evaluate(projections_test, y_test, verbose=1)
score2 = model_unencrypt.evaluate(projections_not_encrypt_test, y_test, verbose=1)
print("\nEncrypt Test score: ", score[0])
print("\nEncrypt Test accuracy: ", score[1])
print("\n\nNot Encrypt Test score: ", score2[0])
print("\nNot Encrypt Test accuracy: ", score2[1])

from keras.models import load_model
model.save('mnist_rsa_rnn.h5')


import matplotlib.pyplot as plt


# history_encrypt = history_encrypt.history
# history_encrypt.keys()
# history = history_not_encrypt.history
# history_not_encrypt.keys()

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