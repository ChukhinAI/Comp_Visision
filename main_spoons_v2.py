import cv2 as cv2

import numpy as np
from skimage import data
import matplotlib.pyplot as plt



image = cv2.imread('original_spoons.jpg')

mask = image < 87
image[mask]=255

plt.plot(image, cmap = 'gray')

plt.show()