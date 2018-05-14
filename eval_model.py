# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:44:34 2018

@author: zaoliu
"""

import numpy as np
import tensorflow as tf
from read_format_input import extract_pickle_file
from read_format_input import reformat

from cnn_model import cnn

pickle_file = 'notMNIST.pickle'

def evaluate():
    graph = tf.Graph()
    batch_size = 50
    locs = './data/'
    with graph.as_default():
        data_path = [locs + 'valid_' + str(i) + '.tfrecords' for i in range(2)]  # address to save the hdf5 file

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(data_path, num_epochs=None)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [28, 28, 1])
        
        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        # Reshape image data into the original shape
        
        
        image_valid, label_valid = tf.train.batch([image, label], batch_size=batch_size,\
        num_threads=2, capacity=5000, enqueue_many=False)
        logits = cnn.inference(image_valid, training = False)
        
        val_prediction = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.cast(tf.argmax(val_prediction, 1), tf.int32), label_valid)
        valid_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    eval_times = 100     
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists('./logs/model2.ckpt'):
            saver.restore(sess, './logs/model2.ckpt')
            print('model resored from saved version')
        valid_accuracies = []    
        for eval_cnt in range(eval_times):
            valid_accuracy_val = sess.run([valid_accuracy])
            valid_accuracies += valid_accuracy_val
            #print('set %d, accuracy = %f' %(eval_cnt, valid_accuracy_val[0]))
        print('average accuracy over %d batches are %f' %(eval_times, np.mean(valid_accuracies)))
        coord.request_stop()
        coord.join(threads)
        print ('Finished validation')
            
        
def main(_):
    evaluate()


if __name__ == '__main__':
    tf.app.run()        