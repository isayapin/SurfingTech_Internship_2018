import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt

model = keras.applications.mobilenet.MobileNet(input_shape=(100, 100, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=None, input_tensor=None, pooling='max', classes=500)
sgd = optimizers.SGD(lr=0.05, decay=0.0002, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])  

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('tiny_dataset/train',
                                                 target_size = (100, 100), 
                                                 batch_size = 550, 
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('tiny_dataset/validation',
                                            target_size = (100, 100),
                                            batch_size = 300,
                                            class_mode = 'categorical')

history = model.fit_generator(
        training_set,
        steps_per_epoch=130,		
        epochs=100,
        validation_data=test_set,
        validation_steps=40)	

model.save('mbnet_model.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/data1/code/training_own_cnn/mobilenet')
plt.show()
