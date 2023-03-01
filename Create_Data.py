import numpy as np
import cv2
import os
from tqdm import tqdm
from random import shuffle

TRAIN_DIR = 'training_set'
TEST_DIR = 'tiny_test'
(IMG_H, IMG_W) = (128, 64)


def create_label(img_name):
    word_label = img_name.split('.')[-3]
    if word_label == 'cat':
        return 1
    elif word_label == 'dog':
        return 0


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_H, IMG_W))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_H, IMG_W))
        testing_data.append([np.array(img_data), create_label(img)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# create_train_data()
# create_test_data()
