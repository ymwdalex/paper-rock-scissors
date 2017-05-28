import numpy as np
from PIL import Image

# Setup up frame and capture area
FRAME_H, FRAME_W = (720, 1280)

CAP_REGION_W = 512
CAP_REGION_H = 512

CAP_REGION_LEFT_RATIO = 0.55
CAP_REGION_TOP_RATIO = 0.15

CAP_REGION_X = int(CAP_REGION_LEFT_RATIO*FRAME_W)
CAP_REGION_Y = int(CAP_REGION_TOP_RATIO*FRAME_H)
CAP_REGION_LEFTTOP = (CAP_REGION_X, CAP_REGION_Y)

CAP_REGION_RIGHT = int(CAP_REGION_X+CAP_REGION_W)
CAP_REGION_BOTTOM = int(CAP_REGION_Y+CAP_REGION_H)
CAP_REGION_RIGHTBOTTOM = (CAP_REGION_RIGHT, CAP_REGION_BOTTOM)

CAP_COLOR = (255,0,0)

PREDICT_TEXT_REGION_X = CAP_REGION_X
PREDICT_TEXT_REGION_Y = int(CAP_REGION_BOTTOM + 0.05*FRAME_W)

MODEL_DICT = {
    'simple': {
        'data_file': 'data',
        'model_file': 'model/temp_cnn.h5',
        'width': 64,
        'height': 64,
    },
    'vgg': {
        'data_file': 'data',
        'model_file': 'model/vgg16_fully_connect_layer.h5',
        'width': 64,
        'height': 64,
    }
}


def resize_from_array(hand_img_array, width, height):
    image = Image.fromarray(hand_img_array)
    image_new_size_array = image.resize([width, height], Image.BILINEAR)
    return np.asarray(image_new_size_array, dtype="int32")