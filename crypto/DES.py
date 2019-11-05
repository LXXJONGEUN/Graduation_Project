from Crypto.Cipher import DES3 as DES
from Crypto.Hash import SHA256 as SHA

class DES_INIT:
    def __init__(self, keytext, ivtext):
        hash = SHA.new() # SHA 객체 생성
        hash.update(bytes(keytext.encode('utf-8'))) # 해시 값 갱신
        keytext = hash.digest() # 해시 값 추출
        self.key = keytext[:24] # 3DES 키 크기 16byte or 24byte

        hash.update(bytes(ivtext.encode('utf-8')))
        ivtext = hash.digest()
        self.iv = ivtext[:8] # 초기화 벡터크기는 8byte

    def encDES(self, plaintext):
        des3 = DES.new(self.key, DES.MODE_CBC, self.iv) #CBC는 초기화 벡터 필요함
        encmsg = des3.encrypt(plaintext)
        return encmsg
    
    def desDES(self, ciphertext):
        des3 = DES.new(self.key, DES.MODE_CBC, self.iv)
        decmsg = des3.decrypt(ciphertext)
        return decmsg


def main():
    keytext = 'kokokoko'
    ivtext = '1234'
    msg = bytes('python36', 'utf-8')

    DEScipher = DES_INIT(keytext, ivtext)
    DES_ENC = DEScipher.encDES(msg)
    DES_DEC = DEScipher.desDES(DES_ENC)

    print("Encryp: ", DES_ENC, "\nDecryp: ", DES_DEC)

if __name__ == '__main__':
    main()