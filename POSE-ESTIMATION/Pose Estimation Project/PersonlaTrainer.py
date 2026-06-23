import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("PoseVideos/video_8.mp4")
#cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    #img = cv2.imread("PoseVideos/img.jpg")
    #img = cv2.resize(img, (620, 1000))
    img = detector.findPose(img, draw = False)
    
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        # # Right arm
        # detector.findAngle(img, 12, 14, 16)
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        
        per = np.interp(angle, (200, 306), (0, 100))
        bar = np.interp(angle, (200, 306), (750, 200))
        #print(angle, per)
        
        # check for the dumbell curl
        if per > 97:
            if dir == 0:
                count += 0.5
                dir = 1
        if per < 5:
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        
        # draw bar
        cv2.rectangle(img, (550, 200), (500, 750), (0, 255, 0), 3)
        cv2.rectangle(img, (550, int(bar)), (500, 750), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (500, 180), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 4)
        
        
        
        # cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(int(count)), (45, 400), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        cv2.putText(img, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (450, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)