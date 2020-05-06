import cv2
from matplotlib import pyplot as plt
# import numpy as np


cells = cv2.imread('cells_v10.png')  # all
cells = cv2.cvtColor(cells, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(cells, (25, 25), cv2.BORDER_DEFAULT)

cv2.imshow("GaussianBlur", img)
cv2.waitKey()

'''
img2 = cv2.blur(cells, (15, 15))
cv2.imshow("blur", img2)
cv2.waitKey()

img3 = cv2.medianBlur(cells, 17)
cv2.imshow("medianBlur", img3)
cv2.waitKey()
'''

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 43, param1=50, param2=20, minRadius=17, maxRadius=42)

# draw
for i in circles[0, :]:
    cv2.circle(cells, (i[0], i[1]), i[2], (164, 68, 166), 2)  # draw	the	outer	circle  # (0, 255, 0)
    cv2.circle(cells, (i[0], i[1]), 2, (0, 255, 0), 3)  # draw	the	center	of	the	circle


print('count: ', circles[0].shape[0])

plt.imshow(cells)
plt.show()


