import numpy as np
import keras
from keras.datasets import mnist
from keras import models, layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM

S = [
    [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
    [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
    [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
    [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
    [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
    [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
    [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
    [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
    [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
    [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
    [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
    [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
    [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
    [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],   
    [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
    [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16],
]

Rcon = [[0x01,0x00,0x00,0x00],
        [0x02,0x00,0x00,0x00],
        [0x04,0x00,0x00,0x00],
        [0x08,0x00,0x00,0x00],
        [0x10,0x00,0x00,0x00],
        [0x20,0x00,0x00,0x00],
        [0x40,0x00,0x00,0x00],
        [0x80,0x00,0x00,0x00],
        [0x1b,0x00,0x00,0x00],
        [0x36,0x00,0x00,0x00]]

def AddRoundKey(state, w):
    w = np.array(w)
    w = w.T
    new_state = []
    
    for i in range(len(state)):
        s = [state[i][j] ^ w[i][j] for j in range(len(state[i]))]
        new_state.append(s)
    
    return new_state

def SubWord(w):
    new_w = [S[int((x/16)%16)][x%16] for x in w]
    return new_w

def xor(a, b):
    s = []
    for i in range(len(a)):
        s.append(a[i]^b[i])
    return s

def LeftRotate(line, n):
    new_line = line[n:]
    for i in range(0, n):
        new_line.append(line[i])
    return new_line

def KeyExpansion(key):
    w = []
    for i in range(0, 4):
        w.append(key[4*i])
        w.append(key[4*i+1])
        w.append(key[4*i+2])
        w.append(key[4*i+3])
    for i in range(4, 44):
        tmp = w[4*(i-1):4*i]
        if i%4==0:
            tmp = xor(SubWord(LeftRotate(tmp,1)), Rcon[int(i/4)-1])
        new_w = xor(w[4*(i-4):4*(i-3)], tmp)
        w = w + new_w
    return w

def SubBytes(state):
    new_s = []
    for k in state:
        row = [S[int((x/16)%16)][x%16] for x in k]
        new_s.append(row)
    return new_s

def positive_Shift(state):
    n = 1
    new_s = []
    new_s.append(state[0])
    for k in state[1:]:
        new_s.append(LeftRotate(k, n))
        n = n+1
    return new_s

def mul(a):
    ashift = (a<<1)%0x100
    if (a&0x80 == 0):
        return ashift
    else:
        return ashift^0x1b

def MixMatrix(state):
    new_state = []
    for i in range(len(state)):
        s = []
        for j in range(len(state[i])):
            r = (mul(state[i%4][j])
                ^ mul(state[(i+1)%4][j]) ^ state[(i+1)%4][j]
                ^ (state[(i+2)%4][j])
                ^ (state[(i+3)%4][j]))
            s.append(r)
        new_state.append(s)
    return new_state

def AES_Encrypt(plain, key):
    plaintext = plain
    #print([hex(x) for x in plaintext])
    plaintext = np.array(plaintext)
    plaintext = plaintext.reshape(4,4)
    
    plaintext = plaintext.T
    key  = key
    w = KeyExpansion(key)
    new_w = []
    
    for i in range(0, int(len(w)/4)):
        new_w.append(w[4*i:4*i+4])
    
    w = new_w
    state = AddRoundKey(plaintext, w[0:4][:])
    
    for i in range(1,10):
        state = SubBytes(state)
        state = positive_Shift(state)
        state = MixMatrix(state)
        state = AddRoundKey(state, w[4*i:4*(i+1)][:])
        
    state = SubBytes(state)
    state = positive_Shift(state)
    state = AddRoundKey(state, w[40:44][:])
    
    Cipher = []
    
    for k in state:
        Cipher += (['0x{:02x}'.format(x) for x in k])
    #print('CipherText: ', Cipher)
    
    return Cipher

key = [0xc9, 0xc9, 0xc9, 0xc9, 0xc9,
       0xc9, 0xc9, 0xc9, 0xc9, 0xc9,
       0xc9, 0xc9, 0xc9, 0xc9, 0xc9, 0xc9]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

projection_x_train = []

for step in range(len(x_train)):
    x_train_image = x_train[step]
    projection_x = [0 for z in range(28)]
    projection_y = [0 for z in range(28)]
    
    for i in range(28):
        for j in range(28):
            if x_train_image[i][j] > 128:
                x_train_image[i][j] = 1
                projection_x[i] += 1
                projection_y[j] += 1
            else:
                x_train_image[i][j] = 0
    
    projection_x = [int(val) for val in projection_x]
    projection_y = [int(val) for val in projection_y]
    projection = []
    projection.append(projection_x)
    projection.append(projection_y)
    projection_x_train.append(projection)

projection_x_test = []

for step in range(len(x_test)):
    x_test_image = x_test[step]
    projection_x = [0 for z in range(28)]
    projection_y = [0 for z in range(28)]
    
    for i in range(28):
        for j in range(28):
            if x_test_image[i][j] > 128:
                x_test_image[i][j] = 1
                projection_x[i] += 1
                projection_y[j] += 1
            else:
                x_test_image[i][j] = 0
    
    projection_x = [int(val) for val in projection_x]
    projection_y = [int(val) for val in projection_y]
    projection = []
    projection.append(projection_x)
    projection.append(projection_y)
    projection_x_test.append(projection)

x_train_encrypt_list=[]

for i in projection_x_train:
    x_train_encrypt_row_list=[]
    for j in i:
        #x_train_encrypt_rowrow_list=[]
        plain_1 = j[0:16]
        plain_2 = j[16:29]
        # numpy array 12개 -> 뒤에 0 4개 padding
        plain_2 = np.pad(plain_2, pad_width=1, mode='constant', constant_values=0)
        plain_2 = np.pad(plain_2, pad_width=1, mode='constant', constant_values=0)
        
        encrypt_1 = AES_Encrypt(plain_1, key)
        encrypt_2 = AES_Encrypt(plain_2, key)
        
        x_train_encrypt_row_list = x_train_encrypt_row_list + [encrypt_1 + encrypt_2[0:12]]
        
    x_train_encrypt_list = x_train_encrypt_list + [x_train_encrypt_row_list]

x_train_encrypt=[]

for i in x_train_encrypt_list:
    x_train_encrypt_row=[]
    for j in i:
        x_train_encrypt_rowrow=[]
        for k in j:
            text = eval(k)
            x_train_encrypt_rowrow = x_train_encrypt_rowrow + [text]
        
        x_train_encrypt_row = x_train_encrypt_row + [x_train_encrypt_rowrow]
    
    x_train_encrypt = x_train_encrypt + [x_train_encrypt_row]

x_test_encrypt_list=[]

for i in projection_x_test:
    x_test_encrypt_row_list=[]
    for j in i:
        #x_train_encrypt_rowrow_list=[]
        plain_1 = j[0:16]
        plain_2 = j[16:29]
        # numpy array 12개 -> 뒤에 0 4개 padding
        plain_2 = np.pad(plain_2, pad_width=1, mode='constant', constant_values=0)
        plain_2 = np.pad(plain_2, pad_width=1, mode='constant', constant_values=0)
        
        encrypt_1 = AES_Encrypt(plain_1, key)
        encrypt_2 = AES_Encrypt(plain_2, key)
        
        x_test_encrypt_row_list = x_test_encrypt_row_list + [encrypt_1 + encrypt_2[0:12]]
        
    x_test_encrypt_list = x_test_encrypt_list + [x_test_encrypt_row_list]

x_test_encrypt=[]

for i in x_test_encrypt_list:
    x_test_encrypt_row=[]
    for j in i:
        x_test_encrypt_rowrow=[]
        for k in j:
            text = eval(k)
            x_test_encrypt_rowrow = x_test_encrypt_rowrow + [text]
        
        x_test_encrypt_row = x_test_encrypt_row + [x_test_encrypt_rowrow]
    
    x_test_encrypt = x_test_encrypt + [x_test_encrypt_row]

from numpy import array
projection_x_train = array(projection_x_train)
projection_x_test = array(projection_x_test)
x_train_encrypt = array(x_train_encrypt)
x_test_encrypt = array(x_test_encrypt)

projection_x_train = projection_x_train.reshape(projection_x_train.shape[0], 2, 28)
projection_x_test = projection_x_test.reshape(projection_x_test.shape[0], 2, 28)
train = x_train_encrypt.reshape(x_train_encrypt.shape[0], 2, 28)
test = x_test_encrypt.reshape(x_test_encrypt.shape[0], 2, 28)

model = Sequential()
model.add(LSTM(120, input_shape=(2,28)))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(projection_x_train, y_train,
                   batch_size=128,
                   epochs=20,
                   verbose=1,
                   validation_split=0.2)

score = model.evaluate(projection_x_test, y_test, verbose=0)
print("손실도: %.2f%%" % (score[0]*100))
print("정확도: %.2f%%" % (score[1]*100))

model_encrypt = Sequential()
model_encrypt.add(LSTM(120, input_shape=(2,28)))
model_encrypt.add(Dense(10))
model_encrypt.add(Activation('softmax'))
model_encrypt.summary()

model_encrypt.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

history_encrypt = model_encrypt.fit(train, y_train,
                                    batch_size=128,
                                    epochs=20,
                                    verbose=1,
                                    validation_split=0.2)

score = model_encrypt.evaluate(test, y_test, verbose=0)
print("손실도: %.2f%%" % (score[0]*100))
print("정확도: %.2f%%" % (score[1]*100))

history_dict = history.history
history_dict.keys()

history_encrypt_dict = history_encrypt.history
history_encrypt_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
loss = history_dict['loss']

encrypt_acc = history_encrypt_dict['acc']
encrypt_loss = history_encrypt_dict['loss']

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

from keras.models import load_model
model.save('mnist_aes_rnn.h5')