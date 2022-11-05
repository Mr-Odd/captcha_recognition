'''
Creator: Odd
Date: 2022-11-04 19:38:06
LastEditTime: 2022-11-05 10:40:59
FilePath: \torch_captcha\generate_captcha.py
Description: Generate captcha
'''
from captcha.image import ImageCaptcha
import os
import random
from tqdm import trange
from params import *

class CaptchaData():
    def __init__(self, total_generate_count=1e5) -> None:
        
        self.charset = charset
        self.max_len = max_len
        self.total_generate_count=total_generate_count
        
    def gen_captcha_text_and_image(self, file_path, width=160, height=60):
        '''
        生成验证码图片文件
        '''
        
        def random_captcha_text(char_set=self.charset[:-1], captcha_size=4, random_len=True):
            '''
            生成验证码的字符
            '''
            if random_len:
                captcha_size = random.randint(4, self.max_len+1)
            captcha_text = []
            for _ in range(captcha_size):
                c = random.choice(char_set)
                captcha_text.append(c)
            return captcha_text
        
        image = ImageCaptcha(width=width, height=height)
        # 获得随机生成的验证码
        captcha_text = random_captcha_text()
        # 把验证码列表转为字符串
        captcha_text = ''.join(captcha_text)
        # 生成验证码
        file_path = os.path.join(os.curdir, file_path)
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        image.write(captcha_text, os.path.join(file_path, captcha_text + '.jpg'))
            
    def generate(self, train_ratio=.6, test_ratio=.2, val_ratio=.2):
            print('---start generate train_dataset---')
            for _ in trange(int(self.total_generate_count*train_ratio)):
                self.gen_captcha_text_and_image('train')
            print('------------finished!------------')
            print()
            print('---start generate test_dataset---')
            for _ in trange(int(self.total_generate_count*test_ratio)):
                self.gen_captcha_text_and_image('test')
            print('------------finished!------------')
            print()
            print('---start generate val_dataset---')  
            for _ in trange(int(self.total_generate_count*val_ratio)):
                self.gen_captcha_text_and_image('val')
            print('------------finished!------------')
        
if __name__ == '__main__':
    # 总生成数
    total_counts = 100000
    cap = CaptchaData(total_generate_count=total_counts)
    cap.generate()