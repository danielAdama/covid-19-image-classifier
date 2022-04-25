import cv2
import os
from config import config
import numpy as np
from face_mask_detector import MaskDetection

mask = MaskDetection(config.PROTOTXT_PATH, config.WEIGHTS_PATH, config.MODELV3, config.IMAGE_PATH)
face_net, mask_net = mask.load_models()
img_list = mask.image_list()

for image in img_list:
    image_name = image.split(os.path.sep)[-1].split('.')[0]
    image_format = image.split(os.path.sep)[-1].split('.')[1]
    print(f"Processing Image: '{image_name}' with format {image_format}")
    image = cv2.imread(image)
    mask.detectAndPredictMask(image, face_net, mask_net)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()