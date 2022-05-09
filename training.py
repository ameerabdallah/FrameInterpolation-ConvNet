import glob
from tensorflow import keras
from preprocess import create_dataset
from model_utils import create_model
import numpy as np

TEST_PATH = '/mnt/f/ConvNet-DataSet/QVI-960/*'
VID_SET_SIZE = 80
DIRS = glob.glob(TEST_PATH)

CROP_SIZE = (384, 384)

def fit_model(model: keras.models.Model, dataset, epochs=5, batch_size=16):
    for img_set in dataset:
        X = np.empty(shape=(len(img_set)-2, CROP_SIZE[0], CROP_SIZE[1], 2), dtype=np.float16)
        Y = np.empty(shape=(len(img_set)-2, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=np.float16)
        for j in range(len(img_set)-2):            
            X[j,:,:,0] = img_set[j] # frame 1
            Y[j,:,:,0] = img_set[j+1] # labeled intermediate frame
            X[j,:,:,1] = img_set[j+2] # frame 3
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)

# just used to print the first 5 by 5 pixels of the image for debugging
def print_image(image):
    for row in image[:5]:
        for pixel in row[:5]:
            print(pixel, end=", ")
        print()
    print('_________')

dataset = create_dataset(DIRS[:VID_SET_SIZE])

model = create_model((None, None, 2))
fit_model(model, dataset, epochs=5)

