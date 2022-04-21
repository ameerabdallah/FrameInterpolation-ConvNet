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

        R.append(src[:,:,2])       #get red pixel values
        G.append(src[:,:,1])       #get green pixel values
        B.append(src[:,:,0])      #get blue pixel values

    return (R,G,B)




#rArray = preprocess("car-turn")[0][0]
#gArray = preprocess("car-turn")[1][0]
#bArray = preprocess("car-turn")[2][0]



