import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model
from numpy import argmax
from PIL import Image

IP =[58,50,42,34,26,18,10,2,
     60,52,44,36,28,20,12,4,
     62,54,46,38,30,22,14,6,
     64,56,48,40,32,24,16,8,
     57,49,41,33,25,17,9,1,
     59,51,43,35,27,19,11,3,
     61,53,45,37,29,21,13,5,
     63,55,47,39,31,23,15,7]
IP_1=[40,8,48,16,56,24,64,32,
        39,7,47,15,55,23,63,31,
        38,6,46,14,54,22,62,30,
        37,5,45,13,53,21,61,29,
        36,4,44,12,52,20,60,28,
        35,3,43,11,51,19,59,27,
        34,2,42,10,50,18,58,26,
        33,1,41,9,49,17,57,25]
# 明文扩展置换
E = [32,1,2,3,4,5,
        4,5,6,7,8,9,
        8,9,10,11,12,13,
        12,13,14,15,16,17,
        16,17,18,19,20,21,
        20,21,22,23,24,25,
        24,25,26,27,28,29,
        28,29,30,31,32,1]

P=[16,7,20,21,29,12,28,17,
    1,15,23,26,5,18,31,10,
    2,8,24,14,32,27,3,9,
    19,13,30,6,22,11,4,25]

PC_1=[57,49,41,33,25,17,9,
    1,58,50,42,34,26,18,
    10,2,59,51,43,35,27,
    19,11,3,60,52,44,36,
    63,55,47,39,31,23,15,
    7,62,54,46,38,30,22,
    14,6,61,53,45,37,29,
    21,13,5,28,20,12,4]
# 密钥置换选择2
PC_2=[14,17,11,24,1,5,3,28,
        15,6,21,10,23,19,12,4,
        26,8,16,7,27,20,13,2,
        41,52,31,37,47,55,30,40,
        51,45,33,48,44,49,39,56,
        34,53,46,42,50,36,29,32]
# S盒
S = [
		# S1
		[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
		 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
		 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
		 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

		# S2
		[[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
		 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
		 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
		 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

		# S3
		[[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
		 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
		 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
		 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

		# S4
		[[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
		 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
		 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
		 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

		# S5
		[[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
		 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
		 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
		 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

		# S6
		[[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
		 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
		 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
		 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

		# S7
		[[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
		 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
		 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
		 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

		# S8
		[[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
		 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
		 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
		 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]],
	]

# 密钥移位次数
LeftRotate=[1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

# 输入明文并得到列表存储的明文的二进制位
def inputText(filename):
    with open(filename,'r')as f:
        text = f.read()
    text = text.split(',')    
    text = [eval(x) for x in text]
    text = ['{:08b}'.format(x) for x in text]
    text = ''.join(text)
    return text

# 对明文进行IP置换
# i位置对应数组索引为i-1
def IP_Transposition(plaintext,index):
    LR = []
    for i in IP:
        LR.append(int(plaintext[i-1+64*(index-1)]))
    L = LR[:32]
    R = LR[32:]
    return L,R
# 逆置换
def IP_reverseTransp(LR):
    tmp = []
    for i in IP_1:
        tmp.append(LR[i-1])
    return tmp
# 输入密钥
def inputKey(s):
    with open(s,'r')as f:
        key = f.read()
    key = key.split(',')
    key = [eval(x) for x in key]
    key = ['{:08b}'.format(x) for x in key]
    key = "".join(key)
    return key

# 密钥置换
def Key_Transposition(key):
    CD = []
    for i in PC_1:
        CD.append(int(key[i-1]))
    C = CD[:28]
    D = CD[28:]
    return C,D

# 密钥循环左移
def Key_LeftRotate(key,n):
    new_key = key[n:]
    for i in range(0,n):
        new_key.append(key[i])
    return new_key


# 密钥压缩
def Key_Compress(C,D):
    key = C+D
    new_key = []
    for i in PC_2:
        new_key.append(key[i-1])
    return new_key

# 明文R扩展为48位
def R_expand(R):
    new_R = []
    for i in E:
        new_R.append(R[i-1])
    return new_R

# 将两列表元素异或
def xor(input1,input2):
    xor_result = []
    for i in range(0,len(input1)):
        xor_result.append(int(input1[i])^int(input2[i]))
    return xor_result

# 将异或的结果进行S盒代替
def S_Substitution(xor_result):
    s_result = []
    for i in range(0,8):
        tmp = xor_result[i*6:i*6+5]
        row = tmp[0]*2+tmp[-1]
        col = tmp[1]*8+tmp[2]*4+tmp[3]*2+tmp[4]
        s_result.append('{:04b}'.format(S[i][row][col]))
    s_result = ''.join(s_result)
    return s_result

# 将S盒代替的结果进行P置换
def P_Transposition(s_result):
    p_result = []
    for i in P:
        p_result.append(int(s_result[i-1]))
    return p_result
# 由列表生成密文
def generateHex(LR):
    result = []
    for i in range(0,8):
        result.append(LR[8*i]*128+LR[8*i+1]*64+LR[8*i+2]*32+LR[8*i+3]*16+LR[8*i+4]*8+LR[8*i+5]*4+LR[8*i+6]*2+LR[8*i+7])
    result = [hex(x) for x in result]
    return result
# F函数
def F(R,K):
    new_R = R_expand(R)
    R_Kxor= xor(new_R,K)
    s_result = S_Substitution(R_Kxor)
    p_result = P_Transposition(s_result)
    return p_result
# 将密文写入文件
def writeFile(filename,Cipher):
    f = open(filename,'w+')
    temp_cipher=''
    for i in range(1,11):
        temp_cipher += ','.join(map(str,Cipher[i-1]))
        if i-1<9:
            temp_cipher += ','
   
    f.write(temp_cipher)
        
        
    
    
# 生成Kset,
def generateKset(key):
    key = inputKey(key)
    C,D = Key_Transposition(key)
    K = []
    for i in LeftRotate:
        C = Key_LeftRotate(C,i)
        C = Key_LeftRotate(D,i)
        K.append(Key_Compress(C,D))
    return K

# 第一轮加密
def DES_encrypt(filename,key):
    # 从文件中读取明文
    plaintext = inputText(filename)
    # 将明文进行置换分离

    Cipher = []
    temp =[]
    for j in range(1,11):
        for index in range(1,99):
            L,R = IP_Transposition(plaintext,index)
            
            # 生成Kset
            K = generateKset(key)
            for i in range(0,15):
                oldR = R
                # F函数
                p_result = F(R,K[i])
                R = xor(L,p_result)
                L = oldR
            p_result = F(R,K[15])
            L = xor(L,p_result)
            reversedP = IP_reverseTransp(L+R)
            temp += generateHex(reversedP)
        Cipher.insert(j,temp)
        temp=[]
 
    return Cipher

model = load_model('./models/mnist_des_cnn.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 28, 28)
y_test = keras.utils.to_categorical(y_test, 10)

xhat_idx = np.random.choice(x_test.shape[0], 10)

xhat = x_test[xhat_idx]

xhat = xhat.reshape(10,784)

xlabel = []
ylabel = []
for i in range(1,11):
    text = [hex(x) for x in xhat[i-1]]
    xlabel += text
    text=[]

xlabel = ','.join(map(str,xlabel))

with open('trainplaintext.txt','w')as f:
    f.write(xlabel)
    f.close()
with open('trainplaintext.txt','r')as f:
    xlabel = f.read()

xlabel = xlabel.split(',')
xlabel = [eval(x) for x in text]

xhat_encrypt=[]

xhat_encrypt = DES_encrypt('trainplaintext.txt','key1.txt')
for i in range(len(xhat_encrypt)):
    for j in range(len(xhat_encrypt[i])):
        xhat_encrypt[i][j] = int(xhat_encrypt[i][j],16)

from numpy import array
xhat_encrypt = array(xhat_encrypt)

x_hat = xhat_encrypt.reshape(xhat_encrypt.shape[0], 28, 28, 1)

yhat = model.predict_classes(x_hat)

images = xhat
images = images.reshape(10, 28, 28)

for i in range(10):
    img = Image.fromarray(images[i], 'L')
    img.save('test_image_' + str(i) + '.png')
    print('True: ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict: ' + str(yhat[i]))