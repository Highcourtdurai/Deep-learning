import cv2
import numpy as np
import sys


def get_output_layers(net):
    
    layer_names=net.getLayerNames()
    
    output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    return output_layers

def draw_prediction(img,class_id,confidence,x,y,x_plus_w,y_plus_h):
    
    label=str(classes(class_id))
    
    color=COLORS[class_id]
    
    cv2.rectangle(img,(x,y),(x_plus_w,y_plus_h),color,2)
    
    cv2.putText(img,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

image=cv2.imread("person.jpg")
width=image.shape[1]
height=image.shape[0]
scale=0.00392

#softmax=np.argmax()

classes=None

with open("coco.names.txt",'r') as f:
    classes=[line.strip() for line in f.readlines()]
    
COLORS=np.random.uniform(0,255,size=(len(classes),3))

net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg.txt")#Creating neural networks

blob=cv2.dnn.blobFromImage(image,scale,(416,416),(0,0,0),True,crop=False)#(0,0,0)-RGB values are zeros,True-swap

net.setInput(blob)

outs=net.forward(get_output_layers(net))

class_ids=[]
confidences=[]
boxes=[]
conf_threshold=0.5
nms_threshold=0.4

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0] *width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
            x=center_x - w/2
            y=center_y - h/2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x,y,w,h])
            
indices=cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)#NMS-Non max supression

for i in indices:
    i=i[0]
    box=boxes[i]
    x=box[0]
    y=box[1]
    w=box[2]
    h=box[3]
    draw_prediction(image,class_ids[i],confidences[i],round(x),round(y),x_plus_w[i],y_plus_h[i])
    
cv2.imshow('object detection',image)
while True:
    key=cv2.waitKey(1)
    if key==27:
        cv2.imwrite("object-detection.jpg",image)
        cv2.destroyAllWindows()
        break
sys.exit()