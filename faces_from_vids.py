from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
import cv2
import json
import shutil
import math
from time import sleep
import timeit

class face_detector(object):
	def __init__(self):
		self.minsize = 100
		self.threshold = [ 0.6, 0.7, 0.7 ]
		self.factor = 0.709
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
			with sess.as_default():
				self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

	def detect(self,img):
		_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		bounding_boxes, landmarks = align.detect_face.detect_face(_img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
		nrof_faces = bounding_boxes.shape[0]

		if nrof_faces > 0:
			det = bounding_boxes[:,0:4]
			x1 = int(det[0,0])
			y1 = int(det[0,1])
			x2 = int(det[0,2])
			y2 = int(det[0,3])
			return det

def image_pro(frame, detector):

	det = detector.detect(frame)

	if det is None:
		thumbnail = False
		return thumbnail

	x1 = int(det[0,0])
	y1 = int(det[0,1])
	x2 = int(det[0,2])
	y2 = int(det[0,3])

	width = x2 - x1
	height = y2 - y1

	if x1 - math.ceil(width/6) > 0:
		x1 -= int(math.ceil(width/6))
	else: x1 = 0

	if x2 + math.ceil(width/6) < cols:
		x2 += int(math.ceil(width/6))
	else: x2 = cols

	if y1 - math.ceil(height/3) > 0:
		y1 -= int(math.ceil(height/3))
	else: y1 = 0

	if y2 + math.ceil(height/16) < rows:
		y2 += int(math.ceil(height/16))
	else: y2 = rows

	thumbnail = frame[y1:y2,x1:x2,:]

	return thumbnail

def video_pro(videoname,detector):

	cmd = 'ffprobe -v quiet -print_format json -show_format -show_streams ' + videoname + '> jason.txt'
	os.system(cmd)
	hjson = json.load(open('jason.txt'))
	rotation = 0
	try:
		rotation =	int(hjson['streams'][0]['tags']['rotate'])
	except KeyError:
		print ('can not find rotate value')

	cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
	cap = cv2.VideoCapture(videoname)
	
	count = 0
	while(1):
		ret, frame = cap.read()
		if ret == False:
			break

		ext_freq = 10
		if count % ext_freq == 0:
			rows,cols = frame.shape[:2]
			M = cv2.getRotationMatrix2D((cols/2,rows/2),abs(360 - rotation),1)
			frame = cv2.warpAffine(frame,M,(cols,rows))

			if image_pro(frame, detector) is False:
				continue
			else:
				pic_loc = k + '/frame%d.jpg'
				cv2.imwrite(pic_loc % count, image_pro(frame, detector))
		count += 1
	cap.release()

if __name__ == '__main__':
	detector = face_detector()
	directory = '.../valid_p'
	classes = ['/hasBMP', '/nonBMP']

	for c in classes:
		ext_folder = directory + c + '_extraction'
		os.mkdir(ext_folder)
		ID_list = os.listdir(directory + c)
		
		for ID in ID_list:
			os.mkdir(ext_folder + '/' + ID)
			ID_dir = directory + c + '/' + ID 
			ID_dir_files = os.listdir(ID_dir)
			ID_path_folders = []
			vid_folders = ['camera', 'ipcamera']
	
			for i in ID_dir_files:
				if i[0:3] == 'ID.':
					ID_image = i

			img = cv2.imread(ID_dir + '/' + ID_image)
			rows,cols = img.shape[:2]
			ID_face = image_pro(img, detector)

			cv2.imwrite(ext_folder + '/' + ID + '/' + ID_image.split('.')[0] + '_face_photo.' + ID_image.split('.')[1], ID_face)

			for j in vid_folders:
				os.mkdir(ext_folder + '/' + ID + '/' + j)
				list = os.listdir(directory + c + '/' + ID + '/' + j)

				for files in list:
					vid_name = directory + c + '/' + ID + '/' + j + '/' + files
					os.rename(vid_name, vid_name.replace(' ', ''))
					files = files.replace(' ', '')
					k = ext_folder + '/' + ID + '/' + j + '/%s' % files.split('.')[0]
					os.mkdir(k)
					video_pro(directory + c + '/' + ID + '/' + j + '/' + files, detector)

