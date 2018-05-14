# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:44:53 2018

@author: zaoliu
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def read_pickle_file(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

def extract_pickle_file(pickle_file, portion = 'valid_'):
    portion_data = portion + 'dataset'
    portion_label = portion + 'labels'
    
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        dataset = save[portion_data]
        labels = save[portion_label]
    
        return dataset, labels

def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels