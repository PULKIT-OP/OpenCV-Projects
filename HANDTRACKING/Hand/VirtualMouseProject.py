import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wScr, hScr = autopy.screen.size()
#print(wScr, hScr)


###########################

wCam, hCam = 640, 480
frameR = 100 # frame reduction
smoothening = 5

#########################

plocx, plocy = 0, 0
clocx, clocy = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    # 1. find the hand landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox= detector.findPosition(img)
    
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)

    
        # 3. check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
    
        cv2.rectangle(img, (frameR, frameR), ((wCam-frameR), hCam-frameR), (255, 0, 255), 2)
        # 4. onlyy index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            
            # 5. convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            
            # 6. smoothen values
            
            clocx = plocx + (x3 - plocx)/smoothening
            clocy = plocy + (y3 - plocy)/smoothening
            # 7. move mouse
            if x3 and y3:
                autopy.mouse.move(clocx, clocy)
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                plocx, plocy = clocx, clocy
    
    # 8. boht index and middle fingers are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            
            # 9. find distance bw fingers
            length, img, lineInfo  = detector.findDistance(8, 12, img)
            print(length)
            
            # 10. click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                
                autopy.mouse.click()
                
    
    
    # 11. frame rate
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # 12. display
    cv2.imshow("Image", img)
    
    cv2.waitKey(1)