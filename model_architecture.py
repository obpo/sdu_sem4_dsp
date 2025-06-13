import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


# ========== [ Selecting Dataset ] ==========
#: Datasets: ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
OUTPUTS = {'byclass':  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
           'bymerge':  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
           'balanced': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt',
           'letters':  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
           'mnist':    '0123456789'}

DATASET = 'byclass'


def create_model(dataset=DATASET):
    model = keras.models.Sequential()

    model.add(layers.Conv2D(6, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(12, (5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(18, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation='sigmoid'))
    model.add(layers.Dense(len(OUTPUTS[dataset]), activation='softmax'))
    return model


def __show(img, label, x, y, offset=0, dataset=DATASET):
    plt.figure(figsize=(15, 10))
    for i in range(x*y):
        plt.subplot(x, y, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img[i], cmap=plt.get_cmap('gray'))
        plt.xlabel(OUTPUTS[dataset][label[i + offset]])
    plt.show()


if __name__ == "__main__":
    print(create_model().summary())