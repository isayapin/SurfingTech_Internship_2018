import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAvgPool2D
from keras.layers import Input,Concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt 

np.random.seed(7)

def SqueezeNet(img_w,img_h,n_channels):
	x = Input(shape=(img_w,img_h,n_channels))
	conv1 = Conv2D(filters=96,kernel_size=(7,7),strides=(2,2),padding='same',activation='relu')(x)
	maxpool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(conv1)
	fire1_squee = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool1)
	fire1_expan1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire1_squee)
	fire1_expan2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire1_squee)
	fire1_out = Concatenate(axis=-1)([fire1_expan1, fire1_expan2])
	fire2_squee = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire1_out)
	fire2_expan1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire2_squee)
	fire2_expan2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire2_squee)
	fire2_out = Concatenate(axis=-1)([fire2_expan1, fire2_expan2])
	fire3_squee = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire2_out)
	fire3_expan1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire3_squee)
	fire3_expan2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire3_squee)
	fire3_out = Concatenate(axis=-1)([fire3_expan1, fire3_expan2])
	maxpool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(fire3_out)	
	fire4_squee = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool2)
	fire4_expan1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire4_squee)
	fire4_expan2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire4_squee)
	fire4_out = Concatenate(axis=-1)([fire4_expan1, fire4_expan2])
	fire5_squee = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire4_out)
	fire5_expan1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire5_squee)
	fire5_expan2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire5_squee)
	fire5_out = Concatenate(axis=-1)([fire5_expan1, fire5_expan2])
	fire6_squee = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire5_out)
	fire6_expan1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire6_squee)
	fire6_expan2 = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire6_squee)
	fire6_out = Concatenate(axis=-1)([fire6_expan1, fire6_expan2])
	fire7_squee = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire6_out)		
	fire7_expan1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire7_squee)
	fire7_expan2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire7_squee)
	fire7_out = Concatenate(axis=-1)([fire7_expan1, fire7_expan2])
	maxpool3 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(fire7_out)
	fire8_squee = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool3)
	fire8_expan1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(fire8_squee)
	fire8_expan2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire8_squee)
	fire8_out = Concatenate(axis=-1)([fire8_expan1, fire8_expan2])
	conv2 = Conv2D(filters=500, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fire8_out)
	Gap = GlobalAvgPool2D(data_format='channels_last')(conv2)
	model = Model(inputs=x,outputs=Gap)
	return model

if __name__ == '__main__':
	model = SqueezeNet(100,100,3)
	sgd = optimizers.SGD(lr=0.05, decay=0.0002, momentum=0.9, nesterov=True) 		#change learning rate
	model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  

	train_datagen = ImageDataGenerator(rescale = 1./255,
		                            shear_range = 0.2,
		                           zoom_range = 0.2,
		                           horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory('/data1/code/training_own_cnn/tiny_dataset/train',
		                                         target_size = (100, 100),
		                                         batch_size = 550, 
		                                         class_mode = 'categorical')

	test_set = test_datagen.flow_from_directory('/data1/code/training_own_cnn/tiny_dataset/validation',
		                                    target_size = (100, 100),
		                                    batch_size = 300,
		                                    class_mode = 'categorical')

	history = model.fit_generator(
		training_set,
		steps_per_epoch=130,		
		epochs=60,
		validation_data=test_set,
		validation_steps=40)

	model.save('/data1/code/training_own_cnn/squeezenet/model_squeeze.h5')

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.savefig('/data1/code/training_own_cnn/squeezenet')
	plt.show()

