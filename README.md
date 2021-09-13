# erfnet_final

## This repository includes the necessary code to implement the ERFNet for lane detection using the Tensorflow 2.5 framework.
## The original code for the ERFNet can be found here: https://github.com/Eromera/erfnet

For Training and Testing  of the network the Tusimple dataset was used. The folder raw_data represents the original structure of the dataset after downloading it. The file tusimple_prprocessing.py can be used to create images and corresponding masks for the task of image segmentation. 

## model.py
Includes the model architecture implemented using TensorFlow.

## training.py
Includes the necessary code to train the network

## predict.py
Includes the necessary code to use a saved model to generate predictions



