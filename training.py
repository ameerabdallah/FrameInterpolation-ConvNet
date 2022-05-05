import glob
from PIL import Image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, Input
import numpy as np
from preprocess import create_dataset, read_images

# read in a jpeg image, seperate the red, green and blue channels and store the image into a 3d numpy array with the shape (1, height, width, 3)

TEST_PATH = '/mnt/f/ConvNet-DataSet/QVI-960'
BATCH_SIZE = 20
DIRS = glob.glob(TEST_PATH + "/0/")

dataset = create_dataset(DIRS[:BATCH_SIZE])
dataset[0]['X'] *= 255

# img = Image.fromarray(dataset[0]['X'][0])
# img.show()

for set in dataset:
    print(set['X'].shape)
    print(set['X'].ndim)
    print(set['X'].astype(np.uint8).dtype)
    img = Image.fromarray(set['X'][0].astype(np.uint8))
    img.save('test_x.jpg')
    img = Image.fromarray(set['Y'][0].astype(np.uint8))
    img.save('test_y.jpg')
    break
    print(set['Y'].shape)
    print(set['Y'].ndim)

exit()
# todo: implement creation of dataset
X = dataset[0]
Y = dataset[1]

model = Sequential()

model.add(Conv3D(filters=64, kernel_size=(2,5,5)), input_shape=(None, None, 2))
model.add(Activation("relu"))

model.compile(loss="", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, batch_size=32, epochs=3, validation_split=0.3)

