###################################################################################################################
# Training Pipeline for the ERFNet                                                                                #
# Author: Maximilian Heinz                                                                                        #
###################################################################################################################

#IMPORT LIBRARIES
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import model
import datetime
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.morphology import label
from tensorflow.keras.preprocessing import image
#from keras_preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import adam_v2
from tensorflow.keras import backend as K
#from keras import backend as K

#SET SOME PARAMETERS
BASE_TRAIN_PATH = '/Path/To/Data/'  # images and labels subfolders!
#Sqared Images ease training configuration 
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 10
IMG_CHANNELS = 3
SEED=42

#-----------------------------------------------------------------------------------------------#
#                          GENERATE TRAINING AND VALIDATION DATA                                #
#-----------------------------------------------------------------------------------------------#

#Get train and mask ids
train_ids = next(os.walk(BASE_TRAIN_PATH + 'images/'))[2]
mask_ids = next(os.walk(BASE_TRAIN_PATH + 'labels/'))[2]

#Prepare train ans mask data-buckets
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# Get and resize train images and masks
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = BASE_TRAIN_PATH + 'images/' + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

for n, id_ in tqdm(enumerate(mask_ids), total=len(train_ids)):
    path = BASE_TRAIN_PATH + 'labels/' + id_
    mask = imread(path)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
    Y_train[n] = mask

print('Done!')

#Creating the training Image and Mask generator
print('Creating the training Image and Mask generator')
image_datagen = image.ImageDataGenerator()
mask_datagen = image.ImageDataGenerator()

print('Flow from Data - Training Set')
x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=SEED)
y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=SEED)

#Creating the validation Image and Mask generator
print('Creating the validation Image and Mask generator')
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

print('Flow from Data - Validation Set')
x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=SEED)
y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=SEED)

# (Optional) Checking if the images fit
# print('# Checking if the images fit')
# imshow(x.next()[0].astype(np.uint8))
# plt.show()
# imshow(np.squeeze(y.next()[0].astype(np.uint8)))
# plt.show()
# imshow(x_val.next()[0].astype(np.uint8))
# plt.show()
# imshow(np.squeeze(y_val.next()[0].astype(np.uint8)))
# plt.show()

#Zip up the generator objects
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

#-----------------------------------------------------------------------------------------------#
#                                        Custom IoU                                             #
#-----------------------------------------------------------------------------------------------#

def iou_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = (K.sum(y_true_f) + K.sum(y_pred_f)) -intersection
    iou = ((intersection + smooth) / (union + smooth))
    return iou

#-----------------------------------------------------------------------------------------------#
#                                        Custom Dice                                             #
#-----------------------------------------------------------------------------------------------#

def dice_coef(y_true, y_pred, smooth =1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#-----------------------------------------------------------------------------------------------#
#                                         Train Model                                           #
#-----------------------------------------------------------------------------------------------#

# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

#Needed to calculate number of steps per epoch after to stop generating objects
NO_OF_TRAINING_IMAGES = len(X_train[:int(X_train.shape[0]*0.9)])
NO_OF_VAL_IMAGES = len(X_train[int(X_train.shape[0]*0.9):])
NO_OF_EPOCHS = 50

# Load Erfnet Model
model = model.get_erfnet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#-----------------------------------------------------------------------------------------------#
#                                      SetUp Hyperparameters                                    #
#                                                                                               #
# - Accuracy Results are reported using Intersection over Union (IoU) metric                    #
# - Training using the Adam Optimization                                                        #
# - Batch Size of 12                                                                            #
# - Momentum of 0.9                                                                             #
# - Weight-Decay of 2e -4                                                                       #
# - Learning Rate Start with 5e-4 and divide by factor 2 every time error becomes stagnant      #        
#-----------------------------------------------------------------------------------------------#

initial_learning_rate = 0.09158 #5e-4\
checkpoint_dir = '/Path/to/directory/'

#Adam Optimizer
opt = Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.9,
    beta_2=0.99,
    decay =2e-4,
    epsilon=1e-08
)

#Compile the Model
model.compile(
    optimizer=opt,
    loss = 'binary_crossentropy',
    metrics=[iou_coef]
)

#Define Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.5,
    patience = 3,
    verbose = 1,
)

checkpoint = ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq='epoch'
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


callbacks = [checkpoint, reduce_lr, tensorboard_callback]

#Fit the Model
model.fit(
    train_generator, 
    epochs=NO_OF_EPOCHS, 
    steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),     
    validation_data=val_generator, 
    validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
    callbacks=callbacks
)
