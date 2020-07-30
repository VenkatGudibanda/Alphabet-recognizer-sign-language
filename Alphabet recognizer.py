
from google.colab import drive
drive.mount('/content/gdrive')

import os
os.mkdir('datasets')

os.mkdir('rajudata')
!unzip -o 'gdrive/My Drive/neeraj.zip' -d datasets

os.mkdir('rajudata1')
!unzip -o 'gdrive/My Drive/neeraj.zip' -d rajudata

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(Dropout(p=0.1))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 26, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasets/neeraj/asl_alphabet_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

#test_set = test_datagen.flow_from_directory('data3/asl_alphabet_test',
                                   #         target_size = (64, 64),
                                #            batch_size = 32,
                                      #      class_mode = 'categorical')
batch_size=32
# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "gdrive/My Drive/best_weights_project.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]



classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = training_set,
                         validation_steps = 32,callbacks=callbacks_list)

a=training_set.class_indices.keys()
a=list(a)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('datasets/neeraj/asl_alphabet_train/C/C10.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image=test_image/255
result =classifier.predict(test_image)
print(a[(np.argmax(result))])

import numpy as np
from keras.preprocessing import image
import glob
filess=glob.glob("G/*.jpg")

filess[0]

for i in filess:
  test_image = image.load_img(i, target_size = (64, 64))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  test_image=test_image/255
  result =classifier.predict(test_image)
  print(a[(np.argmax(result))])



