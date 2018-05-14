# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:50:30 2018

@author: zaoliu
"""
import tensorflow as tf
class cnn(object):
    def __init__():
        return
    #@staticmethod
    def inference(input_layer, training = True):
        
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=16,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
         
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 16]
        # Output Tensor Shape: [batch_size, 14, 14, 16]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 16]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 7, 7, 32]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
        #inception-v1 block 
        conv1x1_bypass = tf.layers.conv2d(inputs = pool2, filters = 20, 
                                   kernel_size = [1, 1], padding = 'same', activation = tf.nn.relu)
        conv1x1_a = tf.layers.conv2d(inputs = pool2, filters = 20, 
                                   kernel_size = [1, 1], padding = 'same', activation = tf.nn.relu)
        conv1x1_b = tf.layers.conv2d(inputs = pool2, filters = 20, 
                                   kernel_size = [1, 1], padding = 'same', activation = tf.nn.relu)
        conv3x3 = tf.layers.conv2d(inputs = conv1x1_a, filters = 20, 
                                   kernel_size = [3, 3], padding = 'same', activation = tf.nn.relu)
        conv5x5 = tf.layers.conv2d(inputs = conv1x1_b, filters = 20, 
                                   kernel_size = [5, 5], padding = 'same', activation = tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=pool2, pool_size=[3, 3], strides=1, padding = 'same')
        conv1x1_c = tf.layers.conv2d(inputs = pool3, filters = 20, 
                                   kernel_size = [1, 1], padding = 'same', activation = tf.nn.relu)
      
        
        incept_out = tf.concat([conv1x1_bypass, conv3x3, conv5x5, conv1x1_c], axis = 3, name = 'concat')
    
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 80]
        # Output Tensor Shape: [batch_size, 7 * 7 * 80]
        incept_out_flat = tf.reshape(incept_out, [-1, 7 * 7 * 80])
    
        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 80]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=incept_out_flat, units=1024, activation=tf.nn.relu)
        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training= training)
    
        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 10]
        logits = tf.layers.dense(inputs=dropout, units=10)
        return logits
    
    def prediction(logits):
        pred_prob = tf.nn.softmax(logits)
        return pred_prob, tf.argmax(pred_prob, axis = 1)
    
    def loss(logits, labels):
        return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))