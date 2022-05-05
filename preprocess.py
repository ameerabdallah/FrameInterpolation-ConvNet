import glob
import cv2
import numpy as np

def preprocess_images(images) -> dict({'X': np.array, 'Y': np.array}):
    return make_x_and_y(images)

def make_x_and_y(images: np.array) -> dict({'X': np.array, 'Y': np.array}):
    # make the x and y values
    if len(images) < 3:
        raise ValueError("Not enough images")

    if len(images) % 2 == 0:
        X = images[:-1]

    X = images[0::2]
    Y = images[1::2]

    assert(len(X)-1 == len(Y))
    return {'X': X, 'Y': Y}

def create_dataset(dirs: str):
    # read images recursively from path
    dataset = []
    for i, dir in enumerate(dirs):
        print("Reading in images from set {}".format(i))
        images = read_images(dir)
        dataset.append(preprocess_images(images))
    return dataset

# read images directly from path
def read_images(path):
    # tuple = (r,g,b)
    # opencv uses BGR

    image_files = glob.glob(path + "/*t*.jpg")

    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    image_shape = first_image.shape

    images = np.empty(shape=(len(image_files), image_shape[0], image_shape[1]), dtype=np.float16)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        images[i] = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return images