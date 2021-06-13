# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# vid=cv2.VideoCapture("samplevideo.mp4")


# while (True):
#     ret,frame=vid.read()
    
#     cv2.imshow("frame",frame)
    
#     if cv2.waitKey(60) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows()


# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os

# cam=cv2.VideoCapture("samplevideo.mp4")

# dataset="datasets"
# name="Durai"

# path=os.path.join(dataset,name)
# if not os.path.isdir(path):
#     os.mkdir(path)

# count=1

# while (count<5):
#     print(count)
#     _,img=cam.read()
#     cv2.imwrite("%s/%s.jpg" %(path,count),img)
#     count+=1
    
#     key=cv2.waitKey(10)
#     if key==27:
#         break

# print("Face Captured successfully")
# cam.release()
# cv2.destroyAllWindows()

# import cv2

# print(cv2.__version__)
# vidcap = cv2.VideoCapture("samplevideo.mp4")
# vidcap.set(cv2.CAP_PROP_POS_MSEC,96000)  
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#   cv2.waitKey(200)
#   count += 1

import cv2

vidcap = cv2.VideoCapture("samplevideo.mp4")
count = 0
success = True
fps = int(vidcap.get(cv2.CAP_PROP_FPS))

while success:
    success,image = vidcap.read()
    print('read a new frame:',success)
    if count%(1*fps) == 0 :
         cv2.imwrite('frame%d.jpg'%count,image)
         print('successfully written 10th frame')
    count+=1
    
