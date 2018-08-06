import numpy as np
import cv2
from math import sqrt


class KalmanFilter:
    def __init__(self, pos):
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        self.mp = np.zeros((2,1), np.float32)
        self.mp[0][0] = pos[0]
        self.mp[1][0] = pos[1]
        self.tp = np.zeros((2,1), np.float32)  

#cap = cv2.VideoCapture('slow_traffic_small.mp4')
#cap = cv2.VideoCapture('mv2_001.avi')  
cap = cv2.VideoCapture('output.mp4')  

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def ROISelection(frame1):
    old_frame = frame1
    r = cv2.selectROI("Image", old_frame, fromCenter=False)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask_image = old_gray.copy()
    start_point = (r[0], r[1])
    end_point = (r[0]+r[2], r[1]+r[3])

    mask_image[:r[1],:] = 0
    mask_image[:,:r[0]] = 0            
    mask_image[r[1]+r[3]:,:] = 0
    mask_image[:,r[0]+r[2]:] = 0

    #old_gray = cv2.bitwise_and(old_gray, old_gray, mask=mask_image)
    #cv2.imshow('lk_track', old_gray)    

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_image, **feature_params)

    ydist = r[0] - int(p0[0][0][0]) 
    xdist = r[1] - int(p0[0][0][1])

    kalman = KalmanFilter((int(p0[0][0][0]), int(p0[0][0][1])))    

    return p0,old_gray,r,ydist,xdist, kalman 


# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

p0, old_gray, r, ydist, xdist, kalman = ROISelection(old_frame)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#r = cv2.selectROI(frame)



while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        kalman.mp[0][0] = p1[0][0][0]
        kalman.mp[1][0] = p1[0][0][1]     
        #print (kalman.mp)   
        kalman.kalman.correct(kalman.mp)
        kalman.tp = kalman.kalman.predict()
        #print (kalman.tp, kalman.tp.shape)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        #for i,(new,old) in enumerate(zip(good_new,good_old)):
        #    a,b = new.ravel()
        #    c,d = old.ravel()
        #    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        point1 = (int(p1[0][0][0]) + ydist, int(p1[0][0][1]) + xdist)
        point2 = (point1[0] + r[2], point1[1] + r[3])
        #print (point1, point2)
        cv2.rectangle(img, point1, point2, color=(0,255,0), thickness=3 )
    else:
        img = frame.copy()

    # kalman.update(measurement)
    # print (kalman.state, img.shape)
    p1 = (kalman.tp[0] + ydist, kalman.tp[1] + xdist)
    p2 = (p1[0] + r[2], p1[1] + r[3])
    cv2.rectangle(img, p1, p2, color=(0,0,255), thickness=3 )    
    #cv2.circle(img, (int(kalman.tp[0]),int(kalman.tp[1])), 15, (255,0,0), 3)

    cv2.imshow('Image', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    if k == ord('c'):
        p0, old_gray, r, ydist, xdist, kalman = ROISelection(frame)
cv2.destroyAllWindows()
cap.release()