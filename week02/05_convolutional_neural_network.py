#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import pickle
import numpy as np


CIFAT_DIR = '../cifar-10-batches-py'
print(os.listdir(CIFAT_DIR))


def load_data(filename):
    """read data from data file"""
    with open(os.path.join(filename), 'rb') as f:
        # data = pickle.load(f, encoding='bytes')

        # Python2.7代码
        data = pickle.load(f)
        return data['data'], data['labels']


class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        # 关于zip函数 具体看
        # http://www.cnblogs.com/frydsh/archive/2012/07/10/2585370.html
        for filename in filenames:
            data, labels = load_data(filename)
            for item, label in zip(data, labels):
                all_data.append(item)
                all_labels.append(label)
        # 关于 vstack函数
        # https://www.cnblogs.com/nkh222/p/8932369.html
        self._data = np.vstack(all_data)
        # 归一化处理
        self._data = self._data / 127.5 - 1;
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # 【0,1,2,3,4】 => [2,1,3,4,0]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch """
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examles')
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAT_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAT_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)
# batch_data, batch_labels = train_data.next_batch(10)
# print(batch_data,batch_labels)


# None 代表输入样本数是不确定的
x = tf.placeholder(tf.float32, [None, 3072])
# None
y = tf.placeholder(tf.int64, [None])

# -1代表缺省 实际上x的维度应该理解为4维
x_image = tf.reshape(x, [-1,3,32,32])
x_image = tf.transpose(x_image, perm = [0,2,3,1])

# 卷积层
# conv1 ：神经元图、feature_map、输出图像
# padding = 'same|valid' same即使用padding
# 32*32
conv1 = tf.layers.conv2d(x_image,
                        32, # 表示输出空间的维数（即卷积过滤器的数量）
                        (3,3),
                        padding = 'same',
                        activation = tf.nn.relu,
                        name = 'conv1')
# 16 * 16
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2), # kernel size
                                   (2,2), # stride 表示卷积的纵向和横向的步长
                                   name = 'pool1')

conv2 = tf.layers.conv2d(pooling1,
                        32,
                        (3,3),
                        padding = 'same',
                        activation = tf.nn.relu,
                        name = 'conv2')
# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2,
                                   (2,2), # kernel size
                                   (2,2), # stride
                                   name = 'pool2')

conv3 = tf.layers.conv2d(pooling2,
                        32,
                        (3,3),
                        padding = 'same',
                        activation = tf.nn.relu,
                        name = 'conv3')

# 4 * 4* 32
pooling3 = tf.layers.max_pooling2d(conv3,
                                   (2,2), # kernel size
                                   (2,2), # stride
                                   name = 'pool3')
# [None,4*4*32]
flatten = tf.layers.flatten(pooling3)



# 这里10表示神经元个数 这个函数就等价于上面写的那么多代码
y_ = tf.layers.dense(flatten, 10)


# y_->softmax
# y -> one_hot
# loss = ylogy_
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)



'''
# [None,10]
p_y_1 = tf.nn.sigmoid(y_)
# 这里-1参数表示缺省值 保证为1列即可
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)
# 计算loss
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
'''

# indices
predict = tf.argmax(y_, 1)
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    # 这里1e-3是学习率 learning rate AdamOptimizer是梯度下降的一个变种
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

'''
到此为止我们的计算图搭建完成
'''

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 1000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, accu_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={x: batch_data, y: batch_labels})
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f,acc: %4.5f' % (i+1, loss_val, accu_val))
        if(i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in xrange(test_steps):
                test_batch_data, test_batch_labels \
                 = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x: test_batch_data,
                        y: test_batch_labels
                    }
                )
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test] Step: %d, acc: %4.5f ' % (i+1, test_acc))

