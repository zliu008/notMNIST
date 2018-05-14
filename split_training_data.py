# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:27:40 2018

@author: zaoliu
"""

from six.moves import cPickle as pickle
import numpy as np
from read_format_input import extract_pickle_file
from read_format_input import reformat
import tensorflow as tf

##the original file notMNIST.pickle is too big
##extract the training portion from it and split it into multiple files to save memory
def split_training_data(numofSplit = 10):
    pickle_file = 'notMNIST.pickle'
    train_dataset, train_labels = extract_pickle_file(pickle_file, portion = 'train_')
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    train_dataset_sections = np.split(train_dataset, numofSplit)
    train_labels_sections = np.split(train_labels, numofSplit)
    for i in range(10):
        fileName = 'notMNIST_train_' + str(i) + '.pickle'
        with open(fileName, 'wb') as f:
            pickle.dump((train_dataset_sections[i], train_labels_sections[i]), f, pickle.HIGHEST_PROTOCOL)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_records(numofSplit = 10, portion = 'valid_'):
    
    pickle_file = 'notMNIST.pickle'
    train_dataset, train_labels = extract_pickle_file(pickle_file, portion = portion)
    train_dataset_sections = np.split(train_dataset, numofSplit)
    train_labels_sections = np.split(train_labels, numofSplit)    
    
    for i in range(numofSplit):
        val_filename = portion + str(i) + '.tfrecords'  # address to save the TFRecords file
        writer = tf.python_io.TFRecordWriter(val_filename)
        for j in range(train_dataset_sections[i].shape[0]):
            # Load the image
            img = train_dataset_sections[i][j].tobytes()
            label = train_labels_sections[i][j]
            # Create a feature
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(img)}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()

def read_tf_records(portion= 'valid_'):
    data_path = [portion + str(i) + '.tfrecords' for i in range(1, 3)]  # address to save the hdf5 file

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
    
    # Cast label data into int32
    label = tf.cast(features['label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [28, 28, 1])
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=50, capacity=5000, 
                                            num_threads=2, min_after_dequeue=500)
        
    return images, labels



def main(_):
    
    create_tf_records(numofSplit = 2)
    create_tf_records(numofSplit = 10, portion = 'train_')
    graph = tf.Graph()
    
    with graph.as_default():
        images, labels = read_tf_records()
    
    
    with tf.Session(graph = graph) as sess:
            tf.global_variables_initializer().run()
            print('Initialized')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            img_val = sess.run([images])
            print(img_val[0].shape)
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()  