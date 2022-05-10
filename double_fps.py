from keras.models import Model, load_model
import cv2
import numpy as np
from model_utils import charbonnier
import glob

model: Model = load_model('model.h5', custom_objects={'charbonnier': charbonnier})
crop_size = (1280, 704)


file_names = glob.glob('F:\\ConvNet-Test_data\\GOPR0384_11_00\\*.png')

file_names = file_names[:180]
frames = np.empty(shape=(len(file_names)-1, crop_size[1], crop_size[0], 2), dtype=np.float32)
prev_image = cv2.imread(file_names[0], cv2.IMREAD_GRAYSCALE)
prev_image = cv2.normalize(prev_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
prev_image = prev_image[:crop_size[1], :crop_size[0]]
cv2.imwrite('F:\\ConvNet-Test_data\\1\\%d.png' % 0, (prev_image*255).astype(np.uint8))
for i, file_name in enumerate(file_names[1:]):
    print("Reading images into memory... {}/{}\r".format(i+1, len(file_names)), end='', flush=i%25==0)
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    image = image[:crop_size[1], :crop_size[0]]
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    frames[i,:,:,0] = prev_image
    frames[i,:,:,1] = image
    prev_image = image
    cv2.imwrite('F:\\ConvNet-Test_data\\1\\%d.png' % (i*2+2), (image*255).astype(np.uint8))

intermediary_frames = model.predict(frames, verbose=1, batch_size=4)

for i, frame in enumerate(intermediary_frames):
    # save frame as image
    cv2.imwrite('F:\\ConvNet-Test_data\\1\\%d.png' % (i*2+1), (frame[:,:,0]*255).astype(np.uint8))

