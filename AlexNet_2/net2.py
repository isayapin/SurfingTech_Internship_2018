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


# Loading the model
model = load_model('/data1/code/training_own_cnn/model_1/model_1.h5')

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
        epochs=10,
        validation_data=test_set,
        validation_steps=70)	

model.save('model_2.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

