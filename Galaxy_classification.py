
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.linalg import polar



class Galaxy_Classification:

    def __init__(self, file_name):

        self.file_name = file_name

    def cv2_read_image(self):

        original_image = cv2.imread(self.file_name)

        return original_image

    def find_center(self, kernel_size=5):

        origininal_img = self.cv2_read_image()
        gray_img = cv2.cvtColor(origininal_img, cv2.COLOR_BGR2GRAY)

        #_,contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #def the center by the max margin value
        gray_x = np.where(gray_img.sum(axis=0) == gray_img.sum(axis=0).max())
        gray_y = np.where(gray_img.sum(axis=1) == gray_img.sum(axis=1).max())
        gray_margin_center = np.array(np.mean(gray_x), np.mean(gray_y))

        #def the center by the max in gray value
        max_x, max_y = np.where(gray_img == gray_img.max())
        gray_max_center = np.array(np.mean(max_x), np.mean(max_y))

        #convolutional center
        kernel = np.ones((kernel_size,kernel_size))
        conv = convolve2d(gray_img, kernel, mode="same")
        conv_max = conv.max()
        conv_x,conv_y = np.where(conv == conv_max)
        conv_center = np.array([np.mean(conv_x), np.mean(conv_y)])

        return conv_center

    def cart2pol(self, x, y, x0=0, y0=0):
        xx, yy = np.meshgrid(x-x0, y-y0)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        temp_phi = np.arctan2(yy, xx)
        phi = np.arctan2(yy, xx)
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                if temp_phi[i][j] < 0:
                    phi[i][j] = temp_phi[i][j] + 2 * np.pi
                else:
                    phi[i][j] = temp_phi[i][j]
        return rho, phi

    def plot_theta_r(self):

        img = self.cv2_read_image()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, np.quantile(img,0.99),
                                    255, cv2.THRESH_BINARY)
        img[thresh == 255] = 0

        _, thresh = cv2.threshold(img, np.mean(img),
                                    255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(thresh)
        sum_up = img.sum()
        ret, thresh = cv2.threshold(img, sum_up/non_zero*0.93,
                                    255, cv2.THRESH_BINARY)

        x, y = img.shape

        center = self.find_center()
        x = np.array(range(x))
        y = np.array(range(y))
        rho, phi = self.cart2pol(x,y,center[0],center[1])

        plt.subplot(projection="polar")
        plt.pcolormesh(phi, rho, thresh)
        plt.plot(phi, rho, color='k', ls='none')
        plt.grid()

        plt.show()








