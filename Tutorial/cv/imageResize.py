import cv2 as cv
import numpy as np

img = cv.imread('res/faces/face1.jpg')

resizeImg = cv.resize(img, dsize=(200,200))

cv.imshow('orig', img)
print('orig-dimension', img.shape)
cv.imshow('resized', resizeImg)
print('resized-dimension', resizeImg.shape)

cv.waitKey(0)

cv.destroyAllWindows()