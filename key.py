#!/usr/bin/python3

# pip install pycryptodomex
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Cipher import AES
from Cryptodome import Random
from binascii import a2b_hex

class aescrypt():
    def __init__(self,text):
        self.text = 'text'
        self.encode_ = 'gbk'
        self.key = self.add_16('neo')
        self.iv =  Random.new().read(AES.block_size)

    def add_16(self,test):
        test = test.encode(self.encode_)
        while len(test) % 16 != 0:
            test += b'\x00'
        return test

    def AES_data_ECB(self,text):
        self.model = AES.MODE_ECB
        self.aes = AES.new(self.key, self.model)  # 创建一个aes对象
        text = self.add_16(text)
        self.encrypt_text = self.aes.encrypt(text)
        result = base64.encodebytes(self.encrypt_text).decode().strip()
        print('加密后的数据:{}'.format(result))
        return result

        
    
    def AES_data_ECB_j(self,text):
        text = base64.decodebytes(text.encode(self.encode_))
        self.decrypt_text = AES.new(self.key, AES.MODE_ECB).decrypt(text)
        data = self.decrypt_text.decode(self.encode_).strip('\0')
        print('解密后的数据:{}'.format(data))
        return data


if __name__ == '__main__':
    text = 'logminerread'
    iv = Random.new().read(AES.block_size)
    key = 'neo'   #密钥
    test = aescrypt('ddd')
    r=test.AES_data_ECB(text)       # 使用AES_ECB加密方式
    test.AES_data_ECB_j(r) 
