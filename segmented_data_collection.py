import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv.VideoCapture(r"nov_update/subject1/static/U/U.mp4")
detector = HandDetector(detectionCon=0.8, maxHands=2)
offset = 125
offset1= 20
imgSize = 300
folder = "nov_update/subject1/static/U/type2"
counter = 0
lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
upper_threshold = np.array([20, 255, 255], dtype=np.uint8)



while True:
    success, imgg = cap.read()
    hands,imgg = detector.findHands(imgg,draw=True)

    if hands:

        hand1 = hands[0]
        #lmList1 = hand1["lmList"]
        x,y,w,h = hand1['bbox']
        cx,cy = hand1['center']
        centerPoint1 = cx,cy
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = imgg[y - offset1:cy + h + offset1, x - offset1:cx + w + offset1]
        imgCropShape = imgCrop.shape


        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            img = imgWhite.copy()
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            skinMask = cv.inRange(img, lower_threshold, upper_threshold)
            skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv.bitwise_and(img, img, mask=skinMask)
            skin = cv.cvtColor(skin, cv.COLOR_HSV2BGR)
            # cv.cvtColor(skin, cv.COLOR_HSV2BGR)
            # crop_img = skin[int(imgSize/2-offset):int(imgSize/2+offset),:]
            # crop_img=skin

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            img = imgWhite.copy()
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            skinMask = cv.inRange(img, lower_threshold, upper_threshold)
            skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv.bitwise_and(img, img, mask=skinMask)
            skin = cv.cvtColor(skin, cv.COLOR_HSV2BGR)
            # cv.cvtColor(skin, cv.COLOR_HSV2BGR)
            # crop_img = skin[:, int(imgSize/2-offset+25):int(imgSize/2+offset+25)]


        if len(hands) == 2:
            hand2 = hands[1]
            #lmList2 = hand2["lmList"]
            x,y,w,h = hand2['bbox']
            cx1,cy1 = hand2['center']
            centerPoint2 = cx1,cy1
            px=(cx1+cx)//2
            py=(cy1+cy)//2
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = imgg[py - offset:py + offset, x - offset:px+ offset]

            imgCropShape = imgCrop.shape


            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                img = imgWhite.copy()
                img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                skinMask = cv.inRange(img, lower_threshold, upper_threshold)
                skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
                skin = cv.bitwise_and(img, img, mask=skinMask)
                skin = cv.cvtColor(skin, cv.COLOR_HSV2BGR)
                # cv.cvtColor(skin, cv.COLOR_HSV2BGR)
                # crop_img = skin[int(imgSize/2-offset):int(imgSize/2+offset),:]
                # crop_img=skin

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                img = imgWhite.copy()
                img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                skinMask = cv.inRange(img, lower_threshold, upper_threshold)
                skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
                skin = cv.bitwise_and(img, img, mask=skinMask)
                skin = cv.cvtColor(skin, cv.COLOR_HSV2BGR)
                # cv.cvtColor(skin, cv.COLOR_HSV2BGR)
                # crop_img = skin[:, int(imgSize/2-offset+25):int(imgSize/2+offset+25)]
        cv.imshow("ImageCrop", imgCrop)
        cv.imshow("ImageWhite", imgWhite)
        

    cv.imshow("Image", imgg)
    if success :
        counter += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg',skin)
        print(counter)
    else:
        break
cap.release()
cap.destroyAllWindows()