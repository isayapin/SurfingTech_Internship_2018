 #coding=utf-8  
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras import optimizers
import numpy as np  
import matplotlib.pyplot as plt
from keras.models import load_model

seed = 7  
np.random.seed(seed)  

model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(128,128,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  	#change rate
model.add(Dense(4096,activation='relu'))  
#model.add(Dropout(0.5))  
model.add(Dense(2000,activation='softmax'))  


sgd = optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('small_dataset/train',
                                                 target_size = (128, 128), #change here
                                                 batch_size = 2048, #
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('small_dataset/validation',
                                            target_size = (128, 128),
                                            batch_size = 1024,
                                            class_mode = 'categorical')

history = model.fit_generator(
        training_set,
        steps_per_epoch=125,		
        epochs=75,
        validation_data=test_set,
        validation_steps=70)	

model.save('/data1/code/training_own_cnn/model_2.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

