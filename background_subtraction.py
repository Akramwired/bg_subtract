import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

#fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv.createBackgroundSubtractorMOG2()
#fgbg = cv.createBackgroundSubtractorKNN()

#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)

    #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgmask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv.destroyAllWindows()