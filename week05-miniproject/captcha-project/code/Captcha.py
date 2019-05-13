# coding=utf-8
from captcha.image import ImageCaptcha
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

class Captcha(object):
    # 所有数字
    number = [chr(i) for i in range(48, 58)]
    # 所有小写字母
    alphabet = [chr(i) for i in range(97, 123)]
    # 所有大写字母
    ALPHABET = [chr(i) for i in range(65, 91)]

    @staticmethod
    def gen_captcha_text_and_image(chars_set, captcha_size=4, is_gen_gray = False, **params):
        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(chars_set)
            captcha_text.append(c)
        captcha_text = ''.join(captcha_text)
        image = ImageCaptcha(**params)
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        if is_gen_gray:
            captcha_image = captcha_image.convert('L')
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image



if __name__ == '__main__':
    random_size_list = [random.randint(50,90) for _ in range(4)]
    text,image = Captcha.gen_captcha_text_and_image(Captcha.number,is_gen_gray = True,width=200,height=100,font_sizes=random_size_list)
    print(text,image,image.shape)
    plt.imshow(image,cmap='gray')
    plt.show()
