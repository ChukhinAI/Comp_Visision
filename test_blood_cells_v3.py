#import cv2
import cv2 as cv2
import numpy as np
import math

# Load Image
img = cv2.imread('donuts.jpg', 0)
# Make copy of original image
cimg2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Find contours
contours = cv2.findContours(255 - img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
print(type(contours))

# Draw all detected contours on image in green with a thickness of 1 pixel
cv2.drawContours(cimg2, contours, -1, color=(0, 255, 0), thickness=1)

# Show the image
cv2.imshow('detected ellipse', cimg2)
cv2.waitKey(0)
cv2.destroyAllWindows()
