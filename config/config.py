import os
import cv2

IMG_SIZE = 112
CHANNEL = 3
BATCH_SIZE = 1
BATCH_SIZE2 = 2
LEARNING_RATE = 1e-4
EPOCH = 20
FONT = cv2.FONT_HERSHEY_COMPLEX
CONF_THRESH = 0.5
DATA_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/dataset/face_mask_dataset'
PROTOTXT_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/my_tensorflow/face_mask/face_detector/deploy.prototxt'
WEIGHTS_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/my_tensorflow/face_mask/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
MODEL_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/my_tensorflow/face_mask/'
IMAGE_PATH = r'/home/daniel/Desktop/programming/pythondatascience/datascience/computerVision/dataset/img_test'
MODELV1 = f"{MODEL_PATH}/{'mask_detectorV1.model'}"
MODELV2 = f"{MODEL_PATH}/{'mask_detectorV2.model'}"
MODELV3 = f"{MODEL_PATH}/{'mask_detectorV3.model'}"

