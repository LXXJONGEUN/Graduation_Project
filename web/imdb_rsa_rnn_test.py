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
from keras.models import load_model
from numpy import argmax
from PIL import Image
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

xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 리뷰 추가
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

review_list = []
for num in xhat:
    review_list.append(decode_review(num))


xhat = pad_sequences(xhat, maxlen=256)

for i, review in enumerate(xhat):
    for j, word in enumerate(review):
        msg = num_padding(word)
        encrypt_word = rsa.encrypt((e, n), msg)
        val = 0
        for k in encrypt_word:
            val += k
        xhat[i][j] = val

model = load_model('./models/imdb_rsa_rnn.h5')
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
    print(review_list[i])