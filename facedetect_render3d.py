'''
https://github.com/small-yellow-duck/timelordart

this script contains two functions: track() and demo()

track() uses python module opencv to take in images from a webcam
(if your laptop doesn't have an internal webcam, bring an external one if you have one)

demo() uses python module vapory to render a cube and sphere and then slowly
rotate the viewer's perspective
'''

import cv2
import sys
from vapory import *
import numpy as np



def track():
	
	use_raw_image = True #don't do any processing on the image from the webcam before sending it to the face recog algo
	
	#cascPathLeft = '/Applications/opencv-2.4.9/data/haarcascades/haarcascade_lefteye_2splits.xml'
	#cascPathLeft = '/Applications/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'
	cascPathFace = '/Applications/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'
	cascPathLeft = '/Applications/opencv-2.4.9/data/haarcascades/haarcascade_mcs_lefteye.xml'
	cascPathRight = '/Applications/opencv-2.4.9/data/haarcascades/haarcascade_mcs_righteye.xml'
	
	cascPath = [cascPathFace]
	
	cascade =[]	
	for i in xrange(len(cascPath)):
		cascade += [cv2.CascadeClassifier(cascPath[i])]
		
	video_capture = cv2.VideoCapture(0)

	while True:
		# Capture frame-by-frame
		
		
		if use_raw_image:
			ret, frame = video_capture.read()
			gray2 = frame
		else:
			ret, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray2 = cv2.equalizeHist(gray)
			#remove saturated regions from the image... this helps if the subject is underlit
			#a = np.array(np.random.randint(200,245, (gray.shape[0], gray.shape[1])),dtype=np.uint8)
			#gray = gray*(gray<245) #+ a*(gray>=245) 
			#gray2 = gray*(gray<150) #+ a*(gray>=245) 
			#gray2 = cv2.equalizeHist(gray)
			
	
		all_faces = []
		for i in xrange(len(cascade)):
			faces = cascade[i].detectMultiScale(
				gray2,
				scaleFactor=4.0,
				minNeighbors=5,
				minSize=(20, 20),
				flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
		
			all_faces += list(faces)
		

		print all_faces, len(all_faces)
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			
		if len(faces) == 2:
			w = 10
			h = 10
			x = (faces[0][0] + faces[0][2] + faces[1][0])/2 -w/2
			y = (faces[0][1] + faces[0][3] + faces[1][1])/2 -h/2
			print x, y
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

		# Display the resulting frame
		cv2.imshow('Video', gray2)
	
		# Stop video capture 
		# in the opencv window (ie NOT in the terminal) hold down a key - you might have to hold it down a bit.
   		print 'click in the opencv window and hold down a key to quit'
		if cv2.waitKey(1) != -1:
			break


	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()
	video_capture.release()



'''
use LightSource to render a cube and a sphere.
rotate the perspective slowly around the axis running from the viewer's left to right
'''

def demo():

	light = LightSource( [2,4,-3], 'color', [1,1,1] )
	sphere1 = Sphere( [0,1,2], 2, Texture( Pigment( 'color', [1,0,1] )))
	sphere2 = Sphere( [0,1,6], 2, Texture( Pigment( 'color', [1,0,0] )))
		
	i = 0	
	while i <30:
		
		#camera = Camera( 'location', [0,2,-3], 'look_at', [0,1,2] )
		camera = Camera( 'location', [0,2+8.0*i/29,-3-4.0*i/29], 'look_at', [0,1,2] )
		

		scene = Scene( camera, objects= [light, sphere1, sphere2])
		
		cv2.imshow('Video', scene.render(width=800, height=600))
		print i, camera
		i += 1
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

	cv2.destroyAllWindows()
