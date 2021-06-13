import numpy as np
import matplotlib.pyplot as plt
import cv2

vid=cv2.VideoCapture("samplevideo.mp4")

while (True):
    ret,frame=vid.read()
    
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
