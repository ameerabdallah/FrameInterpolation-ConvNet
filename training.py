import glob
from tensorflow import keras
from preprocess import create_dataset
from model_utils import create_model

TEST_PATH = '/mnt/f/ConvNet-DataSet/QVI-960/*'
BATCH_SIZE = 20
DIRS = glob.glob(TEST_PATH)

CROP_SIZE = (384, 384)

def fit_model(model, dataset, epochs=5, batch_size=7):
    for i, img_set in enumerate(dataset):
        F1 = img_set[:][i-1]
        F2 = img_set[:][i]
        F3 = img_set[:][i+1]
        model.fit([F1, F3], [F2], epochs=epochs, batch_size=batch_size)

# just used to print the first 5 by 5 pixels of the image for debugging
def print_image(image):
    for row in image[:5]:
        for pixel in row[:5]:
            print(pixel, end=", ")
        print()
    print('_________')

dataset = create_dataset(DIRS[:BATCH_SIZE])
model = create_model((None, None, 2))
keras.models.load_model('model.h5')


model.save('model.h5')

