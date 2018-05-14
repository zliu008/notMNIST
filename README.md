# notMNIST
Tensorflow implementation using tfrecords
1. The CNN model used inception v1 architecture followed by fully connected layers. Drop out is used for regularization. 
2. tfrecords is used to pipeline the data for training and validation. 

Useage:
  1. run split_training_data.py to create tfrecords files.
  2. run train_model.py to train the CNN model. 
