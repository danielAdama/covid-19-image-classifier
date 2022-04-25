import cv2
import os
from config import config
import imutils
from imutils import paths
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import expand_dims


class MaskDetection(object):
    
    """Mask Detection class for detecting faces with Mask, No mask or incorrect mask.
    
    Attributes:
        prototxtPath (string) : representing the Face detector text directory
        weightsPath (string) : representing the Face detector caffe weights directory
        model (string) : representing the path to the Face mask model
        image (string) : representing the path to the test images
        font (integer) : Open-CV Hershey Complex font
        thres (float) : face detection threshold to filter out weak detections

    """
    def __init__(self, prototxtPath=None, weightsPath=None, model=None, img=None, font=config.FONT, thres=config.CONF_THRESH):
        
        self.prototxtPath = prototxtPath
        self.weightsPath = weightsPath
        self.model = model
        self.image = img
        self.font = font
        self.thres = thres
        

    def load_models(self):
        
        """Function to load the Face model and Mask model from the 
        desired directories.
        
        Args:
            None
        Returns:
            deep neural net face model : Face detector model
            conv neural net mask model : Mask detector model
        """

        face_net = cv2.dnn.readNetFromCaffe(self.prototxtPath, self.weightsPath)
        mask_net = load_model(self.model)

        return (face_net, mask_net)

    def image_list(self):

        """Funtion that returns the list of image directories in a list
        """
        return list(paths.list_images(self.image))

    def detectAndPredictMask(self, frame, face_model, mask_model):
        
        """Function that detects the face, process the image and then assigns the 
        corresponding predicitions to the face.
        
        Args:
            frame (image array) : The images as numpy arrays
            face_model (face_net) : face net loaded from load_models method.
            mask_model (mask_net) : mask net loaded from load_models method.

        Returns:
            None
        """

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # Pass the blob through the face network to get the face detections
        face_model.setInput(blob)
        face_detections = face_model.forward()
        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            # Filter out weak detections to ensure the confidence is greater than the minimum
            if confidence > self.thres:
                # Compute the (x, y) coordinates of the bounding box for the object
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                beginX, beginY, stopX, stopY = box.astype('int')
                # Making sure the bounding boxes fall within the dimension of the image
                beginX, beginY = max(0, beginX), max(0, beginY)
                stopX, stopY = min(w - 1, stopX), min(h - 1, stopY)
                # Grab the ROI, convert to RGB ordering, resize and then preprocess
                face = frame[beginY:stopY, beginX:stopX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (112, 112))
                face = img_to_array(face)
                face = preprocess_input(face)
                # Expand to fit the image
                face = expand_dims(face, axis = 0)
                # Model prediction order --> incorrect_mask, with_mask, without_mask
                preds = mask_model.predict(face)[0]
                label = np.argmax(preds)
                
                if label == 0:
                    cv2.putText(frame, f"Incorrect Mask {(preds[0]*100):.2f}%", (beginX-2, beginY-8), self.font, 0.5, (15, 50, 100), 2)
                    cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (15, 50, 100), 2)
                if label == 1:
                    cv2.putText(frame, f"Mask {(preds[1]*100):.2f}%", (beginX-2, beginY-8), self.font, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"No Mask {(preds[2]*100):.2f}%", (beginX-2, beginY-8), self.font, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (beginX, beginY), (stopX, stopY), (0, 0, 255), 2)


# mask = MaskDetection(config.PROTOTXT_PATH, config.WEIGHTS_PATH, config.MODELV3)
# print(mask.img)