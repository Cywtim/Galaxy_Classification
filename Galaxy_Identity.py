

import cv2
import numpy as np

class Galaxy_Identity:

    def __init__(self,file_path):

        self.file_path = file_path

    def galaxy_id(self):

        original_img = cv2.imread(self.file_path)

        gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray_img,
            0.5*np.mean(gray_img), 255,  # if > 130, then change to 255, else to 0
            cv2.THRESH_BINARY)





