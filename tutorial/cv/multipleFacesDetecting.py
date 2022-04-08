
import cv2 as cv
import numpy as np

def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faceDetect = cv.CascadeClassifier('/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    face = faceDetect.detectMultiScale(gray, 1.05)
    for x,y,w,h in face:
        cv.rectangle(img, (x,y), (x+w,y+h), color = (0,0,255), thickness = 2)
    cv.imshow('result', img)

img = cv.imread('res/faces/faces10.jpg')

face_detect_demo(img)

cv.waitKey(0)

cv.destroyAllWindows() 