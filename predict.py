from unicodedata import name
import tensorflow as tf
import os
import sys
import numpy as np
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import model
from tensorflow.python.util.compat import as_str
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.models import load_model
from keras import backend as K


#-----------------------------------------------------------------------------------------------#
#                                        Custom Dice                                            #
#-----------------------------------------------------------------------------------------------#
SMOOTH = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

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

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 10
IMG_CHANNELS = 3

#-----------------------------------------------------------------------------------------------#
#                                          Prediction                                           #
#-----------------------------------------------------------------------------------------------#
BASE_TRAIN_PATH = '/Path/to/TrainData/' 
BASE_TEST_DATA_PATH = "/Path/to/TestData/" 
TEST_PATH_1 = BASE_TEST_DATA_PATH + '0530/'
TEST_PATH_2 = BASE_TEST_DATA_PATH + '0531/'
TEST_PATH_3 = BASE_TEST_DATA_PATH + '0601/'
TEST_PATH_TEST = BASE_TEST_DATA_PATH + 'Test/'



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

#Get Train Ids
test_id_1 = next(os.walk(TEST_PATH_1))[1]
test_id_2 = next(os.walk(TEST_PATH_2))[1]
test_id_3 = next(os.walk(TEST_PATH_3))[1]
test_id_test = next(os.walk(TEST_PATH_TEST))[1]

print(test_id_1)
print(test_id_2)
print(test_id_3)
print(test_id_test)

X_test_1 = np.zeros((len(test_id_1)*20, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test_2 = np.zeros((len(test_id_2)*20, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test_3 = np.zeros((len(test_id_3)*20, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test_test = np.zeros((len(test_id_test)*20, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []
print('Getting and resizing Test images ... ')
sys.stdout.flush()
for id_ in tqdm(enumerate(test_id_test), total=len(test_id_test)):
    path = TEST_PATH_TEST + id_[1]
    if id_[0] == 0:
        for n in range (1,20):
            img = imread(path + '/' + str(n) + '.jpg')[:,:,:IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test_test[n] = img
    elif id_[0] == 1:
        for n in range (1,20):
            img = imread(path + '/' + str(n) + '.jpg')[:,:,:IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test_test[n+19] = img
    elif id_[0] == 2:
        for n in range (1,20):
            img = imread(path + '/' + str(n) + '.jpg')[:,:,:IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test_test[n+39] = img                   


model = load_model('Path/to/saved/model/', custom_objects= {'dice_coef': dice_coef})

preds_test = model.predict(X_test_test, verbose=1)
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

# Threshold predictions

THRESHOLD = 0.5
preds_train_t = (preds_train > THRESHOLD).astype(np.uint8)
preds_val_t = (preds_val > THRESHOLD).astype(np.uint8)
preds_test_t = (preds_test > THRESHOLD).astype(np.uint8)

# Perform a sanity check on some random training samples
iv = random.randint(0, len(preds_train_t))
ia = random.randint(0, len(preds_train_t))
ib = random.randint(0, len(preds_train_t))
img_train = X_train[iv]
img_mask = Y_train[iv]
img_predicted = preds_train_t[iv]
img_train2 = X_train[ia]
img_mask2 = Y_train[ia]
img_predicted2 = preds_train_t[ia]
img_train3 = X_train[ib]
img_mask3 = Y_train[ib]
img_predicted3 = preds_train_t[ib]

figure, axe = plt.subplots(nrows=3, ncols=3)
axe = axe.ravel()

axe[0].imshow(img_train)
axe[0].set_title("Original Test Image 1")

axe[1].imshow(img_mask)
axe[1].set_title("Original Label Image 1")

axe[2].imshow(img_predicted)
axe[2].set_title("Prediction Test Image 1")

axe[3].imshow(img_train2)
axe[3].set_title("Original Test Image 2")

axe[4].imshow(img_mask2)
axe[4].set_title("Original Label Image 2")

axe[5].imshow(img_predicted2)
axe[5].set_title("Prediction Test Image 2")

axe[6].imshow(img_train3)
axe[6].set_title("Original Test Image 3")

axe[7].imshow(img_mask3)
axe[7].set_title("Original Label Image 3")

axe[8].imshow(img_predicted3)
axe[8].set_title("Prediction Test Image 3")

figure.suptitle("Sanity Check with Training Set Images", fontsize=16)
plt.tight_layout()
plt.show()


#Check on Training Samples
ix = 8
iz = 25
iy = 36
iq = 40

img_true = X_test_test[ix]
img_pred = preds_test[ix]
img_true2 = X_test_test[iz]
img_pred2 = preds_test[iz]
img_true3 = X_test_test[iy]
img_pred3 = preds_test[iy]
img_true4 = X_test_test[iq]
img_pred4 = preds_test[iq]

fig, axes = plt.subplots(nrows=2, ncols=4)

ax = axes.ravel()

ax[0].imshow(img_true)
ax[0].set_title("Original Image 1")

ax[1].imshow(img_pred)
ax[1].set_title("Prediction Image 1")

ax[2].imshow(img_true2)
ax[2].set_title("Original Image 2")

ax[3].imshow(img_pred2)
ax[3].set_title("Prediction Image 2")

ax[4].imshow(img_true3)
ax[4].set_title("Original Image 3")

ax[5].imshow(img_pred3)
ax[5].set_title("Prediction Image 3")

ax[6].imshow(img_true4)
ax[6].set_title("Original Image 4")

ax[7].imshow(img_pred4)
ax[7].set_title("Prediction Image 4")

fig.suptitle("Prediction on never seen Test Images", fontsize=16)
plt.tight_layout()
plt.show()
