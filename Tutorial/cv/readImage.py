import cv2 as cv
import numpy as np

img = cv.imread('res/mountain1.jpg')

cv.imshow('read_img', img)

cv.waitKey(0)

cv.destroyAllWindows()