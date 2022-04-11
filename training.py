# ConvNet where input nodes are going to be 2 jpeg images and the output will be the intermediate frame between the two images.
from tkinter import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D, Rescaling, BatchNormalization
from PIL import Image, ImageFilter 
import pickle
import os
import numpy as np
import preprocess

# read in a jpeg image, seperate the red, green and blue channels and store the image into a 3d numpy array with the shape (1, height, width, 3)

dataset_path = './dataset/'

x = 
y = 

x = x/255.0
y = y/255.0

model = Sequential()


model.add(Conv3D(64, (6,5,5)), input_shape = (6, 1590, 910))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size=(2,2)))

model.add(Conv3D(64,(6,5,5)))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, batch_size=32, epochs=3, validation_split=0.3)

