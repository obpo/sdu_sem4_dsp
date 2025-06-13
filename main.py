import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from functools import wraps
from time import time

from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

from model_architecture import create_model, OUTPUTS, DATASET

# ========== [ Logger ] ==========
LOGGING = True
def log(*args, **kwargs):
    if LOGGING:
        print(*args, **kwargs)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.5f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


# ========== [ Selecting Dataset ] ==========
#: Datasets: ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
dataset = DATASET
chars = OUTPUTS[dataset]


# ========== [ Functions ] ==========
def load_model(path="model\\latest.keras"):
    try:
        log('Loading model')
        m = keras.models.load_model(path)
        log('Model loaded')
    except ValueError:
        raise FileNotFoundError(f"No file found at <{path}>")
    log(m.summary())
    return m


def preprocess(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = image/255.0

    if np.mean(image) > 0.5:
        image = 1 - image

    return image


def cut(img, hor_step=6, ver_step=6):

    h, w = img.shape
    x_max, y_max = (w-28)//hor_step+1, (h-28)//ver_step+1
    out = np.zeros((y_max, x_max, 1, 28, 28, 1))

    for y in range(y_max):
        for x in range(x_max):
            i = img[y*ver_step:28+y*ver_step, x*hor_step:28+x*hor_step]
            i = i.reshape((1, 28, 28, 1))
            out[y, x] = i

    return out


def read_image(path, hstep=6, vstep=6, cutoff=0.90, r=2):
    image = cv2.imread(path)
    image = preprocess(image)
    cutImages = cut(image, hor_step=hstep, ver_step=vstep)
    cutImages = np.array(cutImages)

    y, x, *_ = cutImages.shape
    cutImages = cutImages.reshape((y*x, 28, 28, 1))
    predictions = model.predict_on_batch(cutImages)

    text = []

    for i in range(y):
        text.append('')
        for j in range(x):
            c = np.argmax(predictions[i*x+j])
            p = predictions[i*x+j][c]
            out = chars[c] if p > cutoff else ' '
            text[i] += out
        log(f'Line {i+1: 04d} of {y: 04d}: <{text[i]}>')

    return text


def __test():
    testImg720p = cv2.imread('data\\Eliza\\PreprocessedCapture.png')
    test_image = preprocess(cv2.imread('data\\test.png')).reshape((1, 28, 28, 1))

    cutImage = cut(preprocess(testImg720p))

    x, y = 46, 2

    test_image = cutImage[y][x]

    print(cutImage[0][0].shape)
    pred = model.predict(test_image).reshape((len(chars)))
    print(pred)
    sort = np.argsort(pred)[-5:]
    for i in range(5):
        print(f'{chars[sort[4 - i]]}: {pred[sort[4 - i]]}')
    print(sort)

    cv2.imshow('Test Image', test_image.reshape((28, 28)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def __show(img, x, y):
    plt.figure(figsize=(15, 10))
    i = 0
    for xi in range(x):
        for yi in range(y):
            plt.subplot(x, y, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img[(yi)*4+2][(xi)*4+7], cmap=plt.get_cmap('gray'))

            i += 1
    plt.show()


if __name__ == '__main__':
    model = load_model("model\\snapshot_6.keras")

    img_path = 'data\\Eliza\\cropped\\06.png'
    read_image(img_path, vstep=28, hstep=28, cutoff=0.5)
    log('done')