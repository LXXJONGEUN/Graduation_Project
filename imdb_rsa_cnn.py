# INFO ########################################
# 암호화 데이터 타입 : 문자열(IMDB)
# 암호화 알고리즘 : RSA
# 암호화 적용 방식 : 한 리뷰 길이 256으로 padding
# 학습 알고리즘 : CNN(Conv1D: 문자열 1차원)
# 현재 정확도 77%(epoch: 20)
# comment : Embedding을 통해 10000을 100으로 차원 축소
# Test accuracy:  0.77792
###############################################

import keras
import numpy as np
import crypto.RSA2 as rsa
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
imdb = keras.datasets.imdb

def num_padding(data):
        if data < 10:
                msg = '000' + str(data)
        elif data < 100:
                msg = '00' + str(data)
        elif data < 1000:
                msg = '0' + str(data)
        else:
                msg = str(data)
        return msg

# RSA setting#########################
p = 13
q = 23
n = p * q
totient = (p-1)*(q-1)
e = rsa.get_public_key(totient)
d = rsa.get_private_key(e, totient)
######################################


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# 리뷰 단어 개수 256으로 맞춰준다
x_train = pad_sequences(x_train, maxlen=256)
x_test = pad_sequences(x_test, maxlen=256)

x_not_encrypt_train = x_train
x_not_encrypt_test = x_test

print("Encrypt START")
for i, review in enumerate(x_train):
        for j, word in enumerate(review):
                msg = num_padding(word)
                encrypt_word = rsa.encrypt((e, n), msg)
                val = 0
                for k in encrypt_word:
                        val += k
                x_train[i][j] = val


for i, review in enumerate(x_test):
        for j, word in enumerate(review):
                msg = num_padding(word)
                encrypt_word = rsa.encrypt((e, n), msg)
                val = 0
                for k in encrypt_word:
                        val += k
                x_test[i][j] = val

print(x_train.shape)
print(x_test.shape)

#################################################
print("start")
model = Sequential()
model.add(Embedding(input_dim=10000 , output_dim=100 , input_length=256))

model.add(Conv1D(32, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv1D(64, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# model.add(Conv1D(128, kernel_size=3, padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Conv1D(256, kernel_size=3, padding='same'))
# model.add(Activation('relu'))

model.add(Flatten())

# model.add(Dense(10))
# model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

######################################
model_unencrypt = Sequential()
model_unencrypt.add(Embedding(input_dim=10000 , output_dim=100 , input_length=256))

model_unencrypt.add(Conv1D(32, kernel_size=5, padding='same'))
model_unencrypt.add(Activation('relu'))
model_unencrypt.add(Dropout(0.5))

model_unencrypt.add(Conv1D(64, kernel_size=5, padding='same'))
model_unencrypt.add(Activation('relu'))
model_unencrypt.add(Dropout(0.5))

model_unencrypt.add(Flatten())

model_unencrypt.add(Dense(1))
model_unencrypt.add(Activation('sigmoid'))
model_unencrypt.summary()

model.compile(loss='binary_crossentropy',
                optimizer=Adam(), metrics=['accuracy'])

model_unencrypt.compile(loss='binary_crossentropy',
                optimizer=Adam(), metrics=['accuracy'])

history_encrypt = model.fit(x_train, y_train,
                        batch_size=128, epochs=20,
                        verbose=1, validation_split=0.2)

history_not_encrypt = model_unencrypt.fit(x_not_encrypt_train, y_train,
                        batch_size=128, epochs=20,
                        verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=1)
score2 = model_unencrypt.evaluate(x_not_encrypt_test, y_test, verbose=1)
print("\nEncrypt Test score: ", score[0])
print("\nEncrypt Test accuracy: ", score[1])
print("\n\nNot Encrypt Test score: ", score2[0])
print("\nNot Encrypt Test accuracy: ", score2[1])


from keras.models import load_model
model.save('imdb_rsa_cnn.h5')


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