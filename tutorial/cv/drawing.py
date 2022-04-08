import cv2 as cv

img = cv.imread('res/mountain1.jpg')

img = cv.resize(img, dsize=(576,384))

x,y,w,h = 100, 100, 100, 100

cv.rectangle(img,(x,y,x+w, y + h), (0,0,255), thickness=1)
cv.circle(img, center = (x+w, y + h),radius = 100, color=(255,0,0), thickness=2)

cv.imshow('anno-img', img)

cv.waitKey(0)

cv.destroyAllWindows()