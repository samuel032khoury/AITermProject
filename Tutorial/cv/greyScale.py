import cv2 as cv
import numpy as np

img = cv.imread('res/faces/face1.jpg')

greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('grey_img', greyImg)

cv.waitKey(0)

cv.destroyAllWindows()