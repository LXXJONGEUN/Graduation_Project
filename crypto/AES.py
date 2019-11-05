from Crypto.Cipher import AES
from Crypto.Hash import SHA256 as SHA

class AES_init():
    def __init__(self, keytext, ivtext):
        hash = SHA.new()
        hash.update(keytext)

        keytext = hash.digest()
        self.key = keytext[:16] # 128bit = 16byte

        hash.update(ivtext)
        ivtext = hash.digest()
        self.iv = ivtext[:16]

    # 16byte로 짤라야 하므로, 없으면 에러 뜸
    def makeEnable(self, plaintext):
        if(len(plaintext) % 16 != 0):
            plaintext += bytes("#" * (16 - len(plaintext) % 16), "utf-8")
        return plaintext
    
    def encAES(self, plaintext):
        plaintext = self.makeEnable(plaintext)
        aes = AES.new(self.key, AES.MODE_CBC, self.iv)
        encmsg = aes.encrypt(plaintext)
        return encmsg

    def decAES(self, ciphertext):
        aes = AES.new(self.key, AES.MODE_CBC, self.iv)
        decmsg = aes.decrypt(ciphertext)
        return decmsg


def main():
    keytext = bytes('samsjang', 'utf-8')
    ivtext = bytes('1234', 'utf-8')
    msg = bytes('python', 'utf-8')

    AEScipher = AES_init(keytext, ivtext)
    AES_ENC = AEScipher.encAES(msg)
    AES_DEC = AEScipher.decAES(AES_ENC)

    print(AES_ENC)
    print(AES_DEC.decode("utf-8"))
    # bytes to string: 바이트 형태의 자료형 문자열로 변환하기위해 decode("utf-8") 사용

if __name__ == '__main__':
    main()