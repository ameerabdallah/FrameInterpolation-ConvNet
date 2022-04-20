import imghdr
from pickletools import read_decimalnl_long
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from PIL import Image, ImageFilter


def preprocess(path):
    # tuple = (r,g,b)
    # opencv uses bgr

    R = []
    G = []
    B = []

    dir = glob.glob(path + "/*.jpg")

    for image in dir:
        src = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        red_channel = src[:,:,2]        #get red pixel values
        green_channel = src[:,:,1]      #get green pixel values
        blue_channel = src[:,:,0]       #get blue pixel values

        R.append((image, red_channel))
        G.append((image, green_channel))
        B.append((image, blue_channel))

    return (R,G,B)




#im1 = preprocess("car-turn")[0][0]          image path and R 2D Array
#im2 = preprocess("car-turn")[1][0]          image path and G 2D Array

#output image
#im10 = Image.open(preprocess("car-turn")[0][0][0])
#im10 = im10.show()
#im11 = Image.open(preprocess("car-turn")[1][0][0])
#im11 = im11.show()
#im12 = Image.open(preprocess("car-turn")[2][0][0])
#im12 = im12.show()

#2d Arrays
#rArray = preprocess("car-turn")[0][0][1]
#gArray = preprocess("car-turn")[1][0][1]
#bArray = preprocess("car-turn")[2][0][1]
