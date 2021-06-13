import cv2
import os

alg="haarcascade_frontalface_default.xml"

haar=cv2.CascadeClassifier(alg)

cam=cv2.VideoCapture(0)

dataset="datasets"
name="Durai"

path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(200,200)
count=1

while (count<20):
    print(count)
    _,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(grayImg,1.3,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        onlyFace=grayImg[y:y+h,x:x+w]
        Img=cv2.cvtColor(onlyFace,cv2.COLOR_GRAY2BGRA)
        resizeImg=cv2.resize(Img,(width,height))
        
        cv2.imwrite("%s/%s.jpg" %(path,count),resizeImg)
        count+=1
    cv2.imshow("FaceDetection",img)
    key=cv2.waitKey(10)
    if key==27:
        break

print("Face Captured successfully")
cam.release()
cv2.destroyAllWindows()

    
