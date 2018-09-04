#!/usr/bin/python
# -*- coding: utf-8 -*-

# import libs 
from __future__ import division
import requests
from json import JSONDecoder
import os
import timeit

# keys
http_url = 	"https://api-cn.faceplusplus.com/facepp/v3/compare"
key = 		"htSJyFjLlCfw4p1Kn28RIldxNHXSM60G"
secret = 	"MnwCzAMvTugWpBaOgv0C2YMhLWTqZ4ps"

# 					POSITIVES

# getting pairs directories
pos_file = open('/data3/stage_5/facedata/Stage_5/Pairs/random_pairs_facepp/positives.txt', 'r') 
pos_pairs = []
for lines in pos_file.readlines():
	pos_pairs.append(lines.strip('\n'))

# creating image directories from the pair list
directory = '/data1/code/facenet-master/own_data/test/test_160' 		#change

pair_dir_list_1 = []
for i in pos_pairs:
	ID_number = i.split('   ')[0]
	img_loc_1 = directory + '/' + ID_number + '/' + ID_number + '_0001.png'
	img_loc_2 = directory + '/' + ID_number + '/' + ID_number + '_0002.png'
	pair_dir_list_1.append(list([img_loc_1, img_loc_2]))

# running Face++ FaceCompare API
pos_count = 0
count = 0
pos_skipped = 0
for i in pair_dir_list_1:

	count += 1
	print ('Processing positive %d/900' % count) 

	try:	#sometimes files contain one image (instead of two)
		filepath1 = i[0]
		filepath2 = i[1]

		data = {"api_key": key, "api_secret": secret, "return_landmark": "0"}
		files = {"image_file1": open(filepath1, "rb"),"image_file2": open(filepath2, "rb")}

		response = requests.post(http_url, data=data, files=files)
		req_con = response.content.decode('utf-8')
		req_dict = JSONDecoder().decode(req_con)

		threshold = req_dict['thresholds']['1e-3']		
		confidence = req_dict['confidence']			

		if confidence > threshold:
			pos_count += 1

	except Exception: 
		pos_skipped += 1
		continue

print('Skipped %d positive pairs' % pos_skipped, 'Labelled as positive: %d' % pos_count, 'Positive accuracy: %1.5f' % (pos_count/(900 - pos_skipped)))


# 					NEGATIVES

# getting pairs directories
neg_file = open('/data3/stage_5/facedata/Stage_5/Pairs/eval_8/pairs_2/list_worst.txt', 'r')	#change
neg_pairs = []
for lines in neg_file.readlines():
	neg_pairs.append(lines.strip('\n'))

pair_dir_list_2 = []
for i in neg_pairs:
	ID_number_1 = i.split('   ')[0]
	ID_number_2 = i.split('   ')[2]
	img_loc_1 = directory + '/' + ID_number_1 + '/' + ID_number_1 + '_0001.png'
	img_loc_2 = directory + '/' + ID_number_2 + '/' + ID_number_2 + '_0002.png'		#change here depending on wether 1/1 or 1/2 is used as negatives
	pair_dir_list_2.append(list([img_loc_1, img_loc_2]))

# running Face++ FaceCompare API
neg_count = 0
count = 0
neg_skipped = 0
for i in pair_dir_list_2:

	count += 1
	print ('Processing negative %d/900' % count)

	try:
		filepath1 = i[0]
		filepath2 = i[1]

		data = {"api_key": key, "api_secret": secret, "return_landmark": "0"}
		files = {"image_file1": open(filepath1, "rb"),"image_file2": open(filepath2, "rb")}

		response = requests.post(http_url, data=data, files=files)
		req_con = response.content.decode('utf-8')
		req_dict = JSONDecoder().decode(req_con)

		threshold = req_dict['thresholds']['1e-3']		
		confidence = req_dict['confidence']			

		if confidence < threshold:
			neg_count += 1

	except Exception:
		neg_skipped += 1
		continue

print('Skipped %d negative pairs' % neg_skipped, 'Labelled as negative: %d' % neg_count, 'Negative accuracy: %1.5f' % (neg_count/(900 - neg_skipped)))


#					OVERALL ACCURACY

tot_acc = (neg_count + pos_count)/(1800 - pos_skipped - neg_skipped)
print('Accuracy: %f' % tot_acc)

