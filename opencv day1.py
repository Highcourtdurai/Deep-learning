import numpy as np
import matplotlib.pyplot as plt
import cv2

data=cv2.imread("opencv.png")

# cv2.imshow("image",data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(data.shape)

#draw a line 

# start_point=(0,0)
# end_point=(1200,1200)

# color=(0,255,255)
# thickness=3

# image=cv2.line(data,start_point,end_point,color,thickness)

# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Apply text to image

font=cv2.FONT_HERSHEY_SIMPLEX

org=(0,100)

fontScale=1

color=(255,0,0)

thickness=2

image=cv2.putText(data,'openCV',org,font,fontScale,color,thickness,cv2.LINE_AA)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#To convert rgb to gray

gray=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#To convert image in numpy array

data=np.asarray(data)

data1=data/255.0
print(data1)


