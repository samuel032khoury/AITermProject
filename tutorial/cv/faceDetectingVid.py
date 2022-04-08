
import cv2 as cv
from imutils.video import VideoStream

def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faceDetect = cv.CascadeClassifier('/usr/local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    face = faceDetect.detectMultiScale(gray, 1.05)
    for x,y,w,h in face:
        cv.rectangle(img, (x,y), (x+w,y+h), color = (0,0,255), thickness = 2)
    cv.imshow('result', img)

cap = cv.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)

    if ord('q') == cv.waitKey(1):
        break

cv.destroyAllWindows() 
cap.release()