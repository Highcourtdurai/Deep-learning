import numpy as np
import matplotlib.pyplot as plt
import cv2

vid=cv2.VideoCapture("samplevideo.mp4")

while True:
    ret,frame=vid.read()
    
    frame=np.array(frame)
    x=10
    y=10
    h=50
    w=60
    
    frame=frame[x:x+h,y:y+w]
    
    
    
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# while (True):
#     ret,frame=vid.read()
    
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
#     hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hsv-hue,saturation,v-lightness
#     cv2.imshow("frame",hsv)
    
#     if cv2.waitKey(60) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows()