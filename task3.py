import cv2
import numpy as np

img=np.zeros((512,380,3),np.uint8)

clr=(246,234,210)

org=20
x,y=20,20
w,h=100,100
clearence=20

txt=["1","2","3","4","5","6","7","8","9","ok","0","X"]

font=cv2.FONT_HERSHEY_PLAIN
font_size=3
thickness=2

font_clr=(246,234,210)

for i in range(4):
    for j in range(3):
        cv2.rectangle(img,(x,y),(x+w,y+h),clr,2)#2-thickness
        cv2.putText(img,txt[i*3+j], (x+35,y+65),font,3,font_clr,font_size,cv2.LINE_AA)
        
        x=x+w+clearence
        
    y=y+h+clearence
    x=org
       

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# for i in range(4):
#     for j in range(3):
#         print(i," ",j," ",i*3+j)
        