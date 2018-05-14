# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:19:12 2018

@author: zaoliu
"""
import tensorflow as tf
from read_format_input import read_pickle_file
from read_format_input import reformat
from cnn_model import cnn
import numpy as np
from split_training_data import read_tf_records
from eval_model import evaluate

pickle_file = 'notMNIST.pickle'

def train():
    num_steps_to_show_loss = 100
    num_steps_to_check = 500  
    
   
    graph = tf.Graph()
    
    locs = './data/'
    with graph.as_default():
   
        image_batch, label_batch = read_tf_records(locs, portion= 'train_')
        
        #image_test, label_test = tf.train.batch([test_dataset, test_labels], batch_size=50,\
        #num_threads=2, capacity=5000, enqueue_many=True)
        global_step = tf.contrib.framework.get_or_create_global_step()
        logits = cnn.inference(image_batch)
        loss = cnn.loss(logits, label_batch)
        optimizer = tf.train.AdamOptimizer(0.002)
        train_op = optimizer.minimize(loss, global_step=global_step)
        train_prediction = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.cast(tf.argmax(train_prediction, 1), tf.int32), label_batch)
        train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    
    total_epoch = 2000
    losses = []
    best_accuracy = 0
    with tf.Session(graph = graph) as sess:
        tf.global_variables_initializer().run()
        print('Initialized')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists('./logs/model2.ckpt'):
            saver.restore(sess, './logs/model2.ckpt')
            print('model resored from saved version')
        for epoch in range(total_epoch):
            _, loss_val, global_step_val = sess.run([train_op, loss,  global_step])
            losses.append(loss_val)
            if global_step_val % num_steps_to_show_loss == 0:
                print ('step %d, loss = %f' % (global_step_val, np.mean(losses)))
                losses = []
            if global_step_val % num_steps_to_check == 0:
                
                accuracy = train_accuracy.eval()
                print ('step %d, accuracy = %f' %(global_step_val, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    saver.save(sess, './logs/model2.ckpt')
        
        coord.request_stop()
        coord.join(threads)
        print ('Finished')

            
    
def main(_):
    repeat = 0
    maxRepeat = 3
    while repeat < maxRepeat:
        train()
        print('start evaluation...')
        evaluate()
        repeat += 1


if __name__ == '__main__':
    tf.app.run()
