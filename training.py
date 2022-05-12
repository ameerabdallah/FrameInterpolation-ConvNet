import glob
from tensorflow import keras
from preprocess import create_dataset
from model_utils import create_model, charbonnier
import numpy as np
import os
from random import shuffle

model_dir = 'models_1/'
model_name = model_dir+'model.h5'

os.makedirs(model_dir, exist_ok=True)

TEST_PATH = 'F:\\ConvNet-DataSet\\QVI-960\\*'
VID_BATCH_SIZE = 100
DIRS = glob.glob(TEST_PATH)
CROP_SIZE = (384, 384)
EPOCHS = 4

os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

IMGS_PER_SET = 5
def fit_model(model: keras.models.Model, dataset, epochs=5, batch_size=16):
    X = np.empty(shape=(len(dataset)*IMGS_PER_SET, CROP_SIZE[0], CROP_SIZE[1], 2), dtype=np.float32)
    Y = np.empty(shape=(len(dataset)*IMGS_PER_SET, CROP_SIZE[0], CROP_SIZE[1], 1), dtype=np.float32)
    for i, img_set in enumerate(dataset):
        for j in range(len(img_set)-2):            
            X[i*IMGS_PER_SET+j,:,:,0] = img_set[j] # frame 1
            Y[i*IMGS_PER_SET+j,:,:,0] = img_set[j+1] # labeled intermediate frame
            X[i*IMGS_PER_SET+j,:,:,1] = img_set[j+2] # frame 3
    
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

# just used to print the first 5 by 5 pixels of the image for debugging
def print_image(image):
    for row in image[:5]:
        for pixel in row[:5]:
            print(pixel, end=", ")
        print()
    print('_________')

# if the file 'model.h5' exists, load it, otherwise create a new model
model: keras.models.Model = None
if os.path.isfile(model_name):
    model = keras.models.load_model(model_name, custom_objects={'charbonnier': charbonnier})
else:
    model = create_model((None, None, 2))

shuffle(DIRS)

for i in range(len(DIRS)//VID_BATCH_SIZE):
    print("Batch {}".format(i))
    dirs = DIRS[i*VID_BATCH_SIZE:i*VID_BATCH_SIZE+VID_BATCH_SIZE]
    dataset = create_dataset(dirs, CROP_SIZE)
    fit_model(model, dataset, epochs=EPOCHS, batch_size=2)
    if i % 10 == 0 or i < 5:
        model.save(model_name+'.'+str(i))


