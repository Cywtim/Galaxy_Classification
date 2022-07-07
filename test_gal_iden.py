"""

This file is a test to obtain the center of the image in the fold.

"""
import numpy as np
import matplotlib.pyplot as plt
from Galaxy_classification import Galaxy_Classification

file_name = r'C:\\Users\\CHENG\\PycharmProjects' \
            r'\\Galaxy_Classification\\image_000449.jpg'
GC = Galaxy_Classification(file_name)
img = GC.cv2_read_image()
coord = GC.find_center()
#plt.imshow(img)
#plt.scatter(coord[0], coord[1])
#plt.show()
GC.plot_theta_r()

print(coord)
