# coding=utf-8
from Captcha import Captcha
import random
import numpy as np

class ImageData(object):

    @staticmethod
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    @staticmethod
    def text2vec(text):
        text_len = len(text)
        chars_setting_len = len(Captcha.number)
        vector = np.zeros(chars_setting_len * text_len)

        for i,c in enumerate(text):
            # print text
            idx = i * chars_setting_len + ImageData.char2pos(c)
            # print i,CHAR_SET_LEN,char2pos(c),idx
            vector[idx] = 1
        return vector

    @staticmethod
    def gen_captcha_data(batch_size,image_shape,captcha_text_length,captcha_set):
        batch_x = np.zeros([batch_size, image_shape[0]*image_shape[1]])
        batch_y = np.zeros([batch_size, len(captcha_set) * captcha_text_length])
        random_size_list = [random.randint(50,90) for _ in range(captcha_text_length)]
        for i in range(batch_size):
            text, image = Captcha.gen_captcha_text_and_image(captcha_set,captcha_size=captcha_text_length, is_gen_gray=True, width=image_shape[1],
                                                             height=image_shape[0], font_sizes=random_size_list)
            batch_x[i, :] = image.flatten()
            # 这里我们归一化数据
            batch_x[i, :] = batch_x[i, :] / 255
            batch_x[i, :] -= 0.5
            batch_y[i, :] = ImageData.text2vec(text)
        return batch_x,batch_y

    @staticmethod
    def gen_captcha_test( image_shape: tuple, captcha_text_length: int, captcha_set: list ) -> tuple:
        batch_x = np.zeros([image_shape[0] * image_shape[1]])
        random_size_list = [random.randint(50, 90) for _ in range(captcha_text_length)]
        text, image = Captcha.gen_captcha_text_and_image(captcha_set, captcha_size=captcha_text_length,
                                                         is_gen_gray=True, width=image_shape[1],
                                                         height=image_shape[0], font_sizes=random_size_list)
        batch_x = image.flatten()
        # 这里我们归一化数据
        batch_x = batch_x / 255
        batch_x -= 0.5
        return text,batch_x,image


if __name__ == '__main__':
    batch_x, batch_y = ImageData.gen_captcha_data(64, (100,200), 4, Captcha.number)
    print(batch_x[0:5],batch_y[0:5])
