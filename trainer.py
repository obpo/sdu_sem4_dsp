import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from emnist import list_datasets, extract_training_samples, extract_test_samples

from model_architecture import create_model, OUTPUTS, DATASET

# ========== [ Logger ] ==========
LOGGING = True
def log(*args, **kwargs):
    if LOGGING:
        print(*args, **kwargs)


# ========== [ Selecting Dataset ] ==========
#: Datasets: ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
dataset = DATASET
chars = OUTPUTS[dataset]

train_img, train_label = extract_training_samples(dataset)
test_img, test_label = extract_test_samples(dataset)

log(f'Dataset: \'{dataset}\' -> {chars}')
log(f'Training shape:{train_img.shape}, Label shape:{train_label.shape}')
log(f'Testing shape: {test_img.shape}, Label shape:{test_label.shape}\n')

# Normalize: 0,255 -> 0,1
train_img, test_img = train_img/255.0, test_img/255.0

try:
    log('Loading model')
    model = keras.models.load_model("model\\latest.keras")
    log('Model loaded')
except ValueError:
    log('Model not found, creating new model')
    model = create_model()
log(model.summary())


def train(batch_size=64, epochs=5):
    global model
    # loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # optim = keras.optimizers.Adam(lr=0.001)
    metrics = ['accuracy']
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    model.fit(train_img, train_label, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    model.evaluate(test_img, test_label, batch_size=batch_size, verbose=1)


def mainloop():
    global model
    train(batch_size=2048, epochs=25)
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    model.save(f"model\\{now}.keras")
    model.save("model\\latest.keras")
    print(f"\nSaved new model <{now}.keras> as latest")


if __name__ == '__main__':
    while True:
        mainloop()
