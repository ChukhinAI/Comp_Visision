import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.measurements import label
import cv2 as cv2

#TRESHOLD = 190  # original
TRESHOLD = 192
#ROI = np.s_[600:700, 600:700]  # original
ROI = np.s_[400:500, 400:500]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(int)


blood_cells = plt.imread('blood_cells_v2_10_shtuk.jpg')
#blood_cells = plt.imread('blood_cells.jpg')  # original
#blood_cells = plt.imread('spoons_v5.jpg')

#blood_cells = plt.imread('blood_cells_v3.jpg')

blood_cells_gray = rgb2gray(blood_cells)  # original
#blood_cells_gray = cv2.cvtColor(blood_cells, cv2.COLOR_BGR2GRAY)

'''

plt.subplot(221)
plt.title('Grayscale')
#plt.imshow(blood_cells_gray[ROI], cmap='gray')  # original
plt.imshow(blood_cells_gray, cmap='gray')


blood_cells_binary = blood_cells_gray < TRESHOLD
_, num_features = label(blood_cells_binary)
plt.subplot(222)
plt.title(f'Binarized. Cells: {num_features}')
#plt.imshow(blood_cells_binary[ROI], cmap='gray')  # original
plt.imshow(blood_cells_binary, cmap='gray')

blood_cells_binary = binary_erosion(blood_cells_binary)
_, num_features = label(blood_cells_binary)
plt.subplot(223)
plt.title(f'Eroded. Cells: {num_features}')
#plt.imshow(blood_cells_binary[ROI], cmap='gray')  # original
plt.imshow(blood_cells_binary, cmap='gray')
#plt.savefig('eroded.jpg')

blood_cells_binary = binary_dilation(blood_cells_binary)
_, num_features = label(blood_cells_binary)
plt.subplot(224)
plt.title(f'Dilated. Cells: {num_features}')
#plt.imshow(blood_cells_binary[ROI], cmap='gray')  # original
plt.imshow(blood_cells_binary, cmap='gray')

plt.savefig('eroded.jpg', pad_inches=0)
'''

#'''

blood_cells_binary = blood_cells_gray < TRESHOLD
_, num_features = label(blood_cells_binary)


blood_cells_binary = binary_erosion(blood_cells_binary)
_, num_features = label(blood_cells_binary)


plt.axis('off')
margins = {
    "left"   : 10.0,
    "bottom" : 10.0,
    "right"  : 10.0,
    "top"    : 10.0
}
plt.imshow(blood_cells_binary, cmap='gray')
plt.savefig('eroded_10.jpg', pad_inches=0)



#'''

print('type of blood_cells_binary = ', type(blood_cells_binary))

'''

image = cv2.imread('eroded_10.jpg', cv2.IMREAD_GRAYSCALE)
print('type of imread image = ', type(image))

blured = cv2.GaussianBlur(image, (3, 3), 0)

# Находим границы с помощью детектора Кэнни
#edges = cv2.Canny(blured, 100, 200)
#plt.imshow(edges, cmap='gray')

edges = cv2.findContours(cv2.UMat(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5, maxRadius=70)
circles = np.uint16(np.around(circles))

j = 0
for i in circles[0, :]:
    j += 1
    # рисуем внешнюю окружность
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # рисуем центр окружности
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
print("Число монет : ", j)
plt.imshow(image, cmap='gray')
'''


plt.show()
