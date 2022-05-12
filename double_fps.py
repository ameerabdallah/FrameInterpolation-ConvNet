from keras.models import load_model
import cv2
import numpy as np
from model_utils import charbonnier
import os

def double_fps_color(model_path, video_path, output_path):
    model = load_model(model_path, custom_objects={'charbonnier': charbonnier})
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)
    target_fps = fps * 2
    print('Target FPS:', target_fps)
    num_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # add padding to the first frame so that the width and the height are both divisible by 32
    vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2. CAP_PROP_FRAME_WIDTH)))
    
    success, frame_1 = cap.read()
    if not success:
        print("Failed to read first frame")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output = cv2.VideoWriter(output_path, fourcc, target_fps, (vid_size[1], vid_size[0]))
    
    output.write(frame_1)
    
    padding, vid_size_with_padding = get_padding(vid_size)
    
    frame_1 = preprocess_frame(frame_1, padding)
    
    x = np.empty(shape=(3, vid_size_with_padding[0], vid_size_with_padding[1], 2), dtype=np.float32)
    
    count = 1
    
    while success:
        success, frame_3 = cap.read()
        if not success:
            break
        
        print("Processed(frames: {}, seconds: {:.2f}) | {:.3f}/100%\r".format(count, count/fps, (count*100/num_of_frames)), end="")
        
        frame_3_temp = frame_3
        frame_3 = preprocess_frame(frame_3, padding)
        
        x[0,:,:,0] = frame_1[:, :, 0]
        x[1,:,:,0] = frame_1[:, :, 1]
        x[2,:,:,0] = frame_1[:, :, 2]
        
        x[0,:,:,1] = frame_3[:, :, 0]
        x[1,:,:,1] = frame_3[:, :, 1]
        x[2,:,:,1] = frame_3[:, :, 2]
        
        y = np.array(model.predict(x, batch_size=2))
        y = np.moveaxis(np.squeeze(y, axis=3) , 0, -1)
        y = y[padding[0]:vid_size[0]+padding[0], padding[3]:vid_size[1]+padding[3], :] * 255
        y = y.astype(np.uint8)
        
        output.write(y)
        output.write(frame_3_temp)
        
        frame_1 = frame_3
        count += 1
    cap.release()
    output.release()
        
def quadruple_fps_color(model_path, video_path, output_path):
    name, ext = os.path.splitext(output_path)
    double_output_path = name + '_double' + ext
    quadruple_output_path = name + '_quadruple' + ext
    double_fps_color(model_path, video_path, double_output_path)
    double_fps_color(model_path, double_output_path, quadruple_output_path)

def get_padding(vid_size):
    vertical_padding = (vid_size[0] % 32)
    horizontal_padding = (vid_size[1] % 32) 
    
    vertical_padding = 32 - vertical_padding if vertical_padding > 0 else 0
    horizontal_padding = 32 - horizontal_padding if horizontal_padding > 0 else 0
    
    padding = [0, 0, 0, 0]
    
    if vertical_padding % 2 != 0:
        padding[0] = int(np.floor(vertical_padding/2))
        padding[1] = int(np.ceil(vertical_padding/2))
    else:
        padding[0] = int(vertical_padding//2)
        padding[1] = int(vertical_padding//2)
        
    if horizontal_padding % 2 != 0:
        padding[2] = int(np.floor(horizontal_padding/2))
        padding[3] = int(np.ceil(horizontal_padding/2))
    else:
        padding[2] = int(horizontal_padding//2)
        padding[3] = int(horizontal_padding//2)
        
    vid_size_with_padding = (vid_size[0] + vertical_padding, vid_size[1] + horizontal_padding)

    return tuple(padding), vid_size_with_padding

def preprocess_frame(frame, padding):
    
    # add padding
    frame = cv2.copyMakeBorder(frame, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=0)
    
    # normalize
    frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return frame

model_path = "models_1/model.h5.100"
input_video_path = "F:\\Test-Data\\input-15.mp4"
quadruple_fps_color(model_path, input_video_path, "F:\\Test-Data\\output.mp4")