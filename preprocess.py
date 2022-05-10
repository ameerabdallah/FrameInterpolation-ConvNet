import glob
import cv2
import numpy as np

def create_dataset(dirs: str, crop_size=(384, 384)):
    # read images recursively from path
    dataset = []
    for i, dir in enumerate(dirs):
        # print a progress bar
        print("Reading images into memory... {}/{}\r".format(i+1, len(dirs)), end='', flush=i%25==0)
        # print("Reading in images from set {}".format(i))
        img_set = read_images(dir, crop_size)
        # img_set = group_images_in_triples(img_set)
        dataset.append(img_set)
    print("\n")
    return dataset

def read_images(path, crop_size=(384, 384)):
    image_files = glob.glob(path + "/*t*.jpg")

    images = np.empty(shape=(len(image_files), crop_size[0], crop_size[1]), dtype=np.float32)
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