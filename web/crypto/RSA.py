from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

class RSA_INIT:
    def __init__(self):
        self.prkey = RSA.generate(1024) # Private key 1024bit
        self.pbkey = self.prkey.publickey() # Public key

    def encRSA(self, plaintext):
        rsa = PKCS1_OAEP.new(self.pbkey)
        encmsg = rsa.encrypt(plaintext)
        return encmsg

    def desRSA(self, ciphertext):
        rsa = PKCS1_OAEP.new(self.prkey)
        decmsg = rsa.decrypt(ciphertext)
        return decmsg
    

def main():
    msg = '123'
    RSAcipher = RSA_INIT()
    RSA_ENC = RSAcipher.encRSA(bytes(msg, "utf-8"))
    RSA_DEC = RSAcipher.desRSA(RSA_ENC)

    print("Encrpy: ", RSA_ENC, "\nDecryp:", RSA_DEC)

if __name__ == '__main__':
    main()



