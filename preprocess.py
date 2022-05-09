import glob
import cv2
import numpy as np
import tensorflow as tf

# def preprocess_images(images) -> dict({'X': np.array, 'Y': np.array}):
#     return make_x_and_y(images)

# def make_x_and_y(images: np.array) -> dict({'X': np.array, 'Y': np.array}):
#     # make the x and y values
#     if len(images) < 3:
#         raise ValueError("Not enough images")

#     if len(images) % 2 == 0:
#         X = images[:-1]

#     X = images[::2]
#     Y = images[1::2]

#     assert(len(X)-1 == len(Y))
#     return {'X': X, 'Y': Y}

def create_dataset(dirs: str):
    # read images recursively from path
    dataset = []
    for i, dir in enumerate(dirs):
        print("Reading in images from set {}".format(i))
        dataset.append(read_images(dir))
    return dataset

def read_images(path, crop_size=(384, 384)):
    image_files = glob.glob(path + "/*t*.jpg")

    images = np.empty(shape=(len(image_files), crop_size[0], crop_size[1]), dtype=np.float16)
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    crop_data = get_rand_crop(first_image, crop_size)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        image = crop_image(image, crop_data)
        images[i] = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return images

def get_rand_crop(image, crop_size=(384, 384)):
    # randomly crop the image
    height, width = image.shape
    x = np.random.randint(0, height - crop_size[0])
    y = np.random.randint(0, width - crop_size[1])
    result = ((x, y), (x+crop_size[0],y+crop_size[1]))
    return result

def crop_image(image, crop_data):
    return image[crop_data[0][0]:crop_data[1][0], crop_data[0][1]:crop_data[1][1]]