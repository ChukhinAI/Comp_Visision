import sys
import numpy as np
import cv2 as cv

hsv_min = np.array((0, 77, 17), np.uint8)
hsv_max = np.array((208, 255, 255), np.uint8)

if __name__ == '__main__':
    fn = 'donuts.jpg'
    img = cv.imread(fn)

    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )
    thresh = cv.inRange( hsv, hsv_min, hsv_max )
    contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        if len(cnt) > 6:
            ellipse = cv.fitEllipse(cnt)
            cv.ellipse(img, ellipse, (0, 0, 255), 2)

    cv.imshow('contours', img)

    cv.waitKey()
    cv.destroyAllWindows()