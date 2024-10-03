import numpy as np 
import cv2

cap = cv2.VideoCapture("test_case_2/test3.mp4")
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.1, noiseSigma=0.1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
frame_area = cap.get(cv2.CAP_PROP_FRAME_WIDTH)*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
centroid_list = []
frame_counter = 0
roi = (0,0,0,0)

while(1):
    if frame_counter == 10:
        if len(centroid_list) > 2 and centroid_list[0] == centroid_list[-1]:
            roi = rect
        centroid_list = []
        frame_counter = 0
    frame_counter += 1
    _, og_frame = cap.read()
    try:
        frame = cv2.medianBlur(og_frame, 1)
    except cv2.error:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    mask = 255 - fgmask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > frame_area/300 and area < frame_area/2:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            centroid_list.append([(x+w)/2, (y+h)/2])
            if roi != (0,0,0,0):
                x1, y1, w1, h1 = roi
                obj = og_frame[y1:y1+h1, x1:x1+w1]
                blurred = cv2.medianBlur(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY), 5)
                ret, thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
                roi_contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                obj_mask = np.zeros(obj.shape, np.uint8)
                cv2.drawContours(obj_mask, roi_contours, -1, color=(0,250,0), thickness=cv2.FILLED)
                obj[:] = cv2.addWeighted(obj_mask, 0.5, obj, 1, 0)
                cv2.drawContours(obj, roi_contours, -1, color=(0,250,0), thickness=1)
                cv2.putText(og_frame, "Человек оставил предмет", (int(og_frame.shape[1]/2), int(og_frame.shape[0]/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                # cv2.imshow('obj', fgmask)
    cv2.imshow('frame', og_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
cap.release()
cv2.destroyAllWindows()