import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1
global clr, shape
shape = 'rectangle'

#________________________________________________________________________________

# Define a values for B,G,R for shape color
#________________________________________________________________________________

clr = (255,0,255)


# mouse callback function

def draw(event,x,y,flags,param):
    if shape == 'rectangle':draw_rectangle(event,x,y,flags,param)
    else:draw_circle(event,x,y,flags,param)


# function for derawing rectangle 
# It takes the position(pixel) at which the left mouse button is clicked
# and draw a rectangle as the mouse is dragged
# The point at which the left mouse button is released will be 
#   your end points of the rectangle

def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,mode,clr

    if event == cv2.EVENT_LBUTTONDOWN: # checking for if left mouse button clicked 
        drawing = True
        ix,iy = x,y


    elif event == cv2.EVENT_MOUSEMOVE: # checking for if  mouseis dragged
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),clr,-1)
        
    elif event == cv2.EVENT_LBUTTONUP:# checking for if left mouse button released 
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),clr,-1)

#________________________________________________________________________________

#   Write a funnction for drawing a circle using mouse call back event.
#   Use draw_rectangle as reference. 
#   Note your first mouse click should be the first point of Diameter 
#   and the releasing point should be your second point of diameter.  
#________________________________________________________________________________


def draw_circle(event,x,y,flags,param):
     global ix,iy,drawing,mode,clr

     if event == cv2.EVENT_LBUTTONDOWN: # checking for if left mouse button clicked 
         drawing = True
         ix,iy = x,y   #initial position


     elif event == cv2.EVENT_MOUSEMOVE: # checking for if  mouseis dragged
         if drawing == True:
             origin=(int((ix+x)/2),int((iy+y)/2))
             d=((ix-x)**2 +(iy-y)**2)**0.5 #distance between two points#d-diameter
             radius=int(d/2)
             cv2.circle(img,origin,radius,clr,-1)
        
     elif event == cv2.EVENT_LBUTTONUP:# checking for if left mouse button released 
         if drawing== True:
             drawing=False
             origin=(int((ix+x)/2),int((iy+y)/2))
             d=((ix-x)**2 +(iy-y)**2)**0.5 #distance between two points#d-diameter
             radius=int(d/2)
             cv2.circle(img,origin,radius,clr,-1)
            
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    

    if k == ord('r'):
        shape =  'rectangle'
        
    elif k==ord('c'):
        shape= 'circle'
#________________________________________________________________________________

# declare another elif statement for getting key value for circle
#________________________________________________________________________________


    elif k == 27:
        break

cv2.destroyAllWindows()
