import cv2

image=cv2.imread("face.jpg")

image=cv2.resize(image,(300,300))


classifier=cv2.CascadeClassifier("haarcascade_eye.xml")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

coordinates=classifier.detectMultiScale(gray_image)
print(coordinates)

for (x,y,w,h) in coordinates:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    

cv2.imshow("image",image)

while True:
    key=cv2.waitKey(1)
    print(key)
    if key==27:
        cv2.destroyAllWindows()
        break
