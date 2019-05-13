# coding=utf-8
import tensorflow as tf
import numpy as np
from math import ceil

class CNN(object):
    @staticmethod
    def gen_captcha_cnn_structure(image_shape,captcha,w_alpha = 0.01,b_alpha = 0.1):
        if not isinstance(image_shape,tuple) or  not isinstance(captcha,tuple):
            raise Exception('image_shape 和 captache参数必需是一个tuple')
        image_height = 0
        image_width = 0
        if len(image_shape) != 2:
            raise Exception('请先将图片转换为灰度图像')
        if len(captcha) != 2:
            raise Exception('captcha是一个len为2的tuple 第一个参数表示验证码字符个数，第二个参数表示验证码的集合大小')
        captcha_text_len,chars_set_len = captcha
        image_height,image_width = image_shape
        # 这里就直接把彩色图转换为灰度图了
        X = tf.placeholder(tf.float32, [None, image_height * image_width])
        Y = tf.placeholder(tf.float32, [None, chars_set_len * captcha_text_len])
        keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(X, shape=[-1, image_height, image_width, 1])

        # 卷积核大小
        w_cl = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_cl, strides=[1, 1, 1, 1], padding="SAME"), b_c1))
        # 定义一个大小为2x2的maxpooling 步长也为2
        max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 定义一个dropout
        dropout1 = tf.nn.dropout(max_pool1, rate=1 - keep_prob)

        # 卷积核大小
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout1, w_c2, strides=[1, 1, 1, 1], padding="SAME"), b_c2))
        # 定义一个大小为2x2的maxpooling 步长也为2
        max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 定义一个dropout
        dropout2 = tf.nn.dropout(max_pool2, rate=1 - keep_prob)

        # 卷积核大小
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout2, w_c3, strides=[1, 1, 1, 1], padding="SAME"), b_c3))
        # 定义一个大小为2x2的maxpooling 步长也为2
        max_pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 定义一个dropout
        dropout3 = tf.nn.dropout(max_pool3, rate=1 - keep_prob)

        # 定义一个全连接层 这里8*20是这么计算出来的
        i_w = image_width
        i_h = image_height
        # 经过了三次pooling层 每次都是2x2且步长为2的
        for _ in range(3):
            i_w = ceil(i_w / 2.0)
            i_h = ceil(i_h / 2.0)

        # 全连接层 1024个神经元
        w_d = tf.Variable(w_alpha * tf.random_normal([i_w * i_h * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(dropout3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, rate=1 - keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, chars_set_len * captcha_text_len]))
        b_out = tf.Variable(b_alpha * tf.random_normal([chars_set_len * captcha_text_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out, X, Y, keep_prob


    @staticmethod
    def train_cnn(image_shape,captcha_text_len, captcha_set , get_next_batch_func ,acc_need, learning_rate ,**arg_list):
        chars_set_len = len(captcha_set)
        out, X, Y, keep_prob = CNN.gen_captcha_cnn_structure(image_shape,(captcha_text_len, chars_set_len), **arg_list)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=Y))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        predict = tf.reshape(out, [-1, captcha_text_len, chars_set_len])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, captcha_text_len, chars_set_len]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # saver.restore(sess, './model/crack_captcha.model-100')
            sess.run(tf.global_variables_initializer())

            step = 0
            bingo = 0
            while True:
                batch_x, batch_y = get_next_batch_func(128,image_shape,captcha_text_len, captcha_set)
                print(batch_y[0])
                _, loss_,out_ = sess.run([optimizer, loss,out], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                print("第%d次训练 损失值为%.06f 系统输出大小为%s" %(step, loss_, str(out_.shape)))

                if step % 5 == 0:
                    batch_x_test, batch_y_test = get_next_batch_func(100,image_shape,captcha_text_len, captcha_set)
                    acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
                    print(acc)
                    if acc > acc_need:
                        bingo += 1
                        if bingo > 50:
                            saver.save(sess, "./model/crack_captcha.model", global_step=step)
                            break
                step += 1

    @staticmethod
    def test_cnn(batch_x, image,captcha_info,data_file):
        tf.reset_default_graph()
        out, X, Y, keep_prob = CNN.gen_captcha_cnn_structure(image.shape, captcha_info)
        predict = tf.reshape(out, [-1, captcha_info[0], captcha_info[1]])
        max_idx_p = tf.argmax(predict, 2)
        batch_x = batch_x.reshape([-1,X.get_shape().as_list()[-1]])
        saver = tf.train.Saver()
        with tf.Session() as sess:

            saver.restore(sess, data_file)
            # sess.run(tf.global_variables_initializer())
            max_idx_p_ = sess.run([max_idx_p], feed_dict={X:batch_x , keep_prob: 1})

            return ''.join(list(map(lambda item:chr(item+0x30),max_idx_p_[0][0])))

