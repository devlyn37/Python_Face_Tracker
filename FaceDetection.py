import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')




cap = cv2.VideoCapture(0)
frameCount = 0
faceRectangles = []
eyeRectangles = []
k = 0

while 1:
	#keep the image and convert to greyscale
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#takes the image and find objects based on cascade with the parameters being min and max sizes
	faces = face_cascade.detectMultiScale(gray, 2, 5)
	
	#x,y,w,h are rectangle metrics
	for (x,y,w,h) in faces:
		#add to list of rectangles
		faceRectangles.append( [x,y,w,h] )
		#if you want regular facedetection that only displays the current frame
		#then draw face rectangle here
		
		#stores the regions of intrest for detecting eys within
		#no point in seaching outside of the area where faces are
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
        
		#find eyes within face ROI
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			
			eyeRectangles.append( [ex,ey,ew,eh] )
			#drawRectangle around eyes
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,50,255),2)
		
	#Add faces to image
	for faces in faceRectangles:
		cv2.rectangle(img,(faces[0],faces[1]),(faces[0]+faces[2],faces[1]+faces[3]),(255,255,255),2)
	#Add eyes to image
	for eyes in eyeRectangles:
		cv2.rectangle(roi_color,(eyes[0],eyes[1]),(eyes[0]+eyes[2],eyes[1]+eyes[3]),(255,50,255),2)
			
	frameCount = frameCount + 1
	k = k + 1
	if frameCount > 5:
		frameCount = 0
		faceRectangles = []
		eyeRectangles = []
			
	#show the image
	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 60:
		break

cap.release()
cv2.destroyAllWindows()
