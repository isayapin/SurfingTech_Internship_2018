import os
import shutil

# extract the 900 positives images
pos_file = open('.../positives.txt', 'r')
pos_pairs = []
for lines in pos_file.readlines():
	pos_pairs.append(lines.strip('\n'))

directory = '.../facenet-master/own_data/test/test_160'
folder = '.../Pairs/eval_1/positive_pairs'

pair_dir_list_1 = []
for i in pos_pairs:
	ID = i.split('   ')[0]
	shutil.copytree(directory + '/' + ID, folder + '/' + ID)

# extract the 900 negative images
neg_file = open('.../list_worst.txt', 'r')
neg_pairs = []
for lines in neg_file.readlines():
	neg_pairs.append(lines.strip('\n'))

# creating image directories from the pair list
directory = '.../facenet-master/own_data/test/test_160'

pair_dir_list_2 = []
for i in neg_pairs:
	ID_number_1 = i.split('   ')[0]
	ID_number_2 = i.split('   ')[2]
	img_loc_1 = directory + '/' + ID_number_1 + '/' + ID_number_1 + '_0001.png'
	img_loc_2 = directory + '/' + ID_number_2 + '/' + ID_number_2 + '_0002.png'
	pair_dir_list_2.append(list([img_loc_1, img_loc_2]))

#print(pair_dir_list_2)

count = 1
for i in pair_dir_list_2:
	pair_dir = '/data3/stage_5/facedata/Stage_5/Pairs/eval_13/worst_pairs_pics/%d' % count
	os.mkdir(pair_dir)
	shutil.copy(i[0], pair_dir)
	shutil.copy(i[1], pair_dir)
	count += 1
