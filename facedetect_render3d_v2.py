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

import time
from math import pi

from OCC.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf
from OCC.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.TopLoc import TopLoc_Location
from OCC.Display.SimpleGui import init_display

import sys, select, os


def build_shape(display):
	boxshp = BRepPrimAPI_MakeBox(50., 50., 50.).Shape()
	ais_boxshp = display.DisplayShape(boxshp, update=True)
	return ais_boxshp
	
	
#initialize the display	
#display, start_display, add_menu, add_function_to_menu = init_display()	
def track_render(display):

	display.EraseAll()
	ais_boxshp = build_shape(display)
	ax1 = gp_Ax1(gp_Pnt(25, 25, 25), gp_Dir(0., 0., 1.))
	aCubeTrsf = gp_Trsf() 
	angle = 0.0
	tA = time.time()
	
	
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

	angle = 0.0
	while True:
		# Capture frame-by-frame
		
		
		if use_raw_image:
			ret, frame = video_capture.read()
			gray2 = frame
		else:
			ret, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#gray2 = cv2.equalizeHist(gray)
			#remove saturated regions from the image... this helps if the subject is underlit
			#a = np.array(np.random.randint(200,245, (gray.shape[0], gray.shape[1])),dtype=np.uint8)
			gray = gray*(gray<200) #+ a*(gray>=245) 
			#gray2 = gray*(gray<150) #+ a*(gray>=245) 
			gray2 = cv2.equalizeHist(gray)
			
	
		all_faces = []
		for i in xrange(len(cascade)):
			'''
			faces = cascade[i].detectMultiScale(
				gray2,
				scaleFactor=4.0,
				minNeighbors=5,
				minSize=(20, 20),
				flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
			'''
			
			faces = cascade[i].detectMultiScale(
				gray2,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30),
				flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
	
		
			all_faces += list(faces)
		

		#print all_faces, len(all_faces)
		print all_faces
		size_largest =0
		for f in all_faces:
			if f[2]*f[3] > size_largest:
				size_largest = f[2]*f[3]
				face = f
			
		
		'''
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			

		# Display the resulting frame
		cv2.imshow('Video', gray2)
		'''
		
		aCubeTrsf.SetRotation(ax1, angle)
		aCubeToploc = TopLoc_Location(aCubeTrsf)
		display.Context.SetLocation(ais_boxshp, aCubeToploc)
		display.Context.UpdateCurrentViewer()
		try:
			angle -= (face[0] -  all_faces_prev[0])/300.0
		except:
			None	
		all_faces_prev = face
	
		# Stop video capture 
		# in the opencv window (ie NOT in the terminal) hold down a key - you might have to hold it down a bit.
   		#print 'click in the opencv window and hold down a key to quit'
		if cv2.waitKey(1) != -1:
			break
			
			
		
		




	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()
	video_capture.release()

	
	

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
			#gray2 = cv2.equalizeHist(gray)
			#remove saturated regions from the image... this helps if the subject is underlit
			#a = np.array(np.random.randint(200,245, (gray.shape[0], gray.shape[1])),dtype=np.uint8)
			gray = gray*(gray<200) #+ a*(gray>=245) 
			#gray2 = gray*(gray<150) #+ a*(gray>=245) 
			gray2 = cv2.equalizeHist(gray)
			
	
		all_faces = []
		for i in xrange(len(cascade)):
			'''
			faces = cascade[i].detectMultiScale(
				gray2,
				scaleFactor=4.0,
				minNeighbors=5,
				minSize=(20, 20),
				flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
			'''
			
			faces = cascade[i].detectMultiScale(
				gray2,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30),
				flags=cv2.cv.CV_HAAR_SCALE_IMAGE
			)
	
		
			all_faces += list(faces)
		

		print all_faces, len(all_faces)
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			

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
		
		print dir(scene)
		cv2.imshow('Video', scene.render(width=800, height=600))
		print i, camera
		i += 1
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	

	cv2.destroyAllWindows()






#display, start_display, add_menu, add_function_to_menu = init_display()
def occ(display):

 	'''
	display, start_display, add_menu, add_function_to_menu = init_display()
	my_box = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
 
	display.DisplayShape(my_box, update=True)
	#start_display()
	'''
	

	display.EraseAll()
	ais_boxshp = build_shape(display)
	ax1 = gp_Ax1(gp_Pnt(25., 25., 25.), gp_Dir(0., 0., 1.))
	aCubeTrsf = gp_Trsf() 
	angle = 0.0
	tA = time.time()
	n_rotations = 200
	for i in range(n_rotations):
		aCubeTrsf.SetRotation(ax1, angle)
		aCubeToploc = TopLoc_Location(aCubeTrsf)
		display.Context.SetLocation(ais_boxshp, aCubeToploc)
		display.Context.UpdateCurrentViewer()
		angle += 2*pi / n_rotations
	print("%i rotations took %f" % (n_rotations, time.time() - tA))
	
	
	
	
	