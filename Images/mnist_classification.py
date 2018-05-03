#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:48:21 2018

@author: jie
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

###1. placeholders for data (x) and labels (y)
x = tf.placeholder(tf.float32, shape=[None, 784]) # 28*28=784
y = tf.placeholder(tf.float32, shape=[None, 10])  # 10 digits

x_input = tf.reshape(x, [-1, 28, 28, 1])





###2.define layers
######first convolutional layer
# 32 5x5 filters with stride of 1, ReLU activation
conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[5,5], activation=tf.nn.relu)

# first pooling layer
# 2x2 max pooling with stride of 2
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=[2,2])

# second convolutional layer
conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=[5,5], activation=tf.nn.relu)

# second pooling layer
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2])

# flatten the final feature maps
flat = tf.layers.flatten(pool2)

# pass flattened input into the first fully connected layer
fc1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

# define second fully connected layer for 0-9 digit classification
y_pred = tf.layers.dense(inputs=fc1, units=10) 

# output probabilities of input image belonging to each digit class
probabilities = tf.nn.softmax(y_pred)



###3. loss function and accuracy
# calculate mean cross entropy over entire batch of samples. 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction=tf.argmax(y_pred,1)



####4. 
tf.summary.scalar('loss',cross_entropy) 
tf.summary.scalar('acc',accuracy)

merged_summary_op = tf.summary.merge_all() #combine into a single summary which we can run on Tensorboard










