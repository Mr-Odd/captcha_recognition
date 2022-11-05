'''
Creator: Odd
Date: 2022-11-05 10:39:56
LastEditTime: 2022-11-05 10:40:24
FilePath: \torch_captcha\params.py
Description: 
'''
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                    'v', 'w', 'x', 'y', 'z']

# 总字符数
charset = numbers + alphabets + ['<pad>', '<eos>']
# 最大验证码长度
max_len = 6