from __future__ import print_function
#"""
#Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
#"""
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data_folder = os.path.abspath('.')  # current path
print(data_folder)

# data path
alpha_data_filename = os.path.join(data_folder,"alpha_norm.mat")
print(alpha_data_filename)

alpha_label_filename = os.path.join(data_folder,"alpha_label.mat")
print(alpha_label_filename)

# read data from matlab format
alpha_data = sio.loadmat(alpha_data_filename)
alpha_label = sio.loadmat(alpha_label_filename)

# data: subject_data*number_of_subjects
alpha_data_array = np.array(alpha_data['alpha_norm'],dtype=np.float32)
# label data, there are 2 classes in the data
alpha_label_array = np.array(alpha_label['alpha_label'],dtype=np.float32)
# transpose, because my data shape is (subject_data*number_of_subjects)
alpha_data_2d_T = alpha_data_array.T

from sklearn.model_selection import train_test_split
# split the data, because my dataset is very small, just 108 subjects,so I just set the (test_size=0.1)
X_train, X_test, y_train, y_test = train_test_split(alpha_data_2d_T, alpha_label_array, random_state=14, test_size=0.1)

import tensorflow as tf

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

def max_pool_2x2_n(x):
    # stride [1, x_movement, y_movement, 1]
    # different strides (strides=[1,2,2,1])
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 5476])   # 74*74
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)  # dropout probability
x_image = tf.reshape(xs, [-1, 74, 74, 1])
# print(x_image.shape)  # [n_samples, 74,74,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,96]) # patch 5*5, in size 1, out size 96
b_conv1 = bias_variable([96])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 74x74x96
h_pool1 = max_pool_2x2(h_conv1)                          # output size 74x74x96

## conv2 layer ##
W_conv2 = weight_variable([5,5, 96, 128]) # patch 5*5, in size 96, out size 128
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 74*74*128
h_pool2 = max_pool_2x2_n(h_conv2)                        # output size 37*37*128

## full connection 1 layer
W_fc1 = weight_variable([37*37*128, 512])
b_fc1 = bias_variable([512])
# [n_samples, 37, 37, 128] ->> [n_samples, 37*37*128]
h_pool2_flat = tf.reshape(h_pool2, [-1, 37*37*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## full connection 2 layer
W_fc2 = weight_variable([512, 1])
b_fc2 = bias_variable([1])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# note: tf.initialize_all_variables() no long valid from 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train, keep_prob: 1})
    if i % 50 == 0:
        print(compute_accuracy(
            X_train,y_train))


