import os
import shutil
import random

pic_folder = '.../hasBMP_extraction'
test_folder = '.../raw_12'
os.mkdir(test_folder)

ID_list = os.listdir(pic_folder)

for ID in ID_list:
	
	os.mkdir(test_folder + '/' + ID)
	
	#photo1 09_free
	list_free = os.listdir(pic_folder + '/' + ID + '/camera/09_free')
	name_1 = random.choice(list_free)
	photo_1 = pic_folder + '/' + ID + '/camera/09_free/' + name_1
	shutil.copy(photo_1, test_folder + '/' + ID)
	os.rename(test_folder + '/' + ID + '/' + name_1, test_folder + '/' + ID + '/' + ID + '_0001.jpg')
	
	#photo2 01_left MIDDLE
	list_left = os.listdir(pic_folder + '/' + ID + '/camera/01_left')
	for i in range (0, len(list_left)):
		list_left[i] = list([int(list_left[i].split('.')[0][5:]), list_left[i]])

	list_left.sort()
	mid = int(round(len(list_left)/2))
	name_2 = list_left[mid][1]
	photo_2 = pic_folder + '/' + ID + '/camera/01_left/' + name_2
	shutil.copy(photo_2, test_folder + '/' + ID)	
	os.rename(test_folder + '/' + ID + '/' + name_2, test_folder + '/' + ID + '/' + ID + '_0002.jpg')
