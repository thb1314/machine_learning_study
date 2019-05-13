# coding=utf-8
from ImageData import ImageData
from CNN import CNN
from Captcha import Captcha
import matplotlib.pyplot as plt

CNN.train_cnn((100,50),2, Captcha.number , get_next_batch_func = ImageData.gen_captcha_data ,acc_need=0.98, learning_rate=0.01)
# 测试下
'''
acc = 0
for _ in range(100):
    text,batch_x,image = ImageData.gen_captcha_test((100,50),2,Captcha.number)
    # plt.imshow(image,cmap="gray")
    # plt.show()
    predict = CNN.test_cnn(batch_x, image,(2, 10),  "./model/crack_captcha.model-2290")

    acc += (predict == text)
print(acc)
'''
