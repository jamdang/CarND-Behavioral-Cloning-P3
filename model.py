import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
samples = []
with open('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)
print("number of samples: ", len(train_samples))

#print("steering?  ",train_samples[0][3])

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

from keras.layers import Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.5
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center  = 'data1/IMG/'+batch_sample[0].split('/')[-1]
                name_left    = 'data1/IMG/'+batch_sample[1].split('/')[-1]
                name_right   = 'data1/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                left_image   = cv2.imread(name_left)
                right_image  = cv2.imread(name_right)
                #image_flipped = np.fliplr(center_image)
                #angle_flipped = -center_angle
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                #images.append(image_flipped)
                angles.append(center_angle)
                angles.append(center_angle+correction)
                angles.append(center_angle-correction)
                #angles.append(angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            #print("X_train shape 1: ",X_train.shape)
            #X_train = X_train[:,80:,:,:] 
            y_train = np.array(angles)
			
            #print("X_train shape: ",X_train.shape)
            #print("y_train shape: ",y_train.shape)
			
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#####################
# model construction
#####################

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
ch, row, col = 3, 90, 320  # Trimmed image format
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))

# Add a conv layer
nb_filters  = 24
kernel_size = (5,5)
pool_size = (2, 2)
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(row, col, ch)))
# Add an activation layer
model.add(Activation('relu'))
# Add a pooling layer
model.add(MaxPooling2D(pool_size=pool_size))
# Add dropout
#model.add(Dropout(0.5))

nb_filters  = 36
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],activation='relu'))
# Add a pooling layer
model.add(MaxPooling2D(pool_size=pool_size))
# Add dropout
model.add(Dropout(0.5))

nb_filters  = 48
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],activation='relu'))
# Add a pooling layer
model.add(MaxPooling2D(pool_size=pool_size))
# Add dropout
#model.add(Dropout(0.5))

kernel_size = (3,3)
nb_filters  = 64
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],activation='relu'))
# Add dropout
#model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],activation='relu'))
# Add dropout
model.add(Dropout(0.5))

# Add a flatten layer
model.add(Flatten())
# Add a fully connected layer
model.add(Dense(100, init='normal', activation='tanh'))
# Add dropout
model.add(Dropout(0.5))

# Add a fully connected layer
model.add(Dense(50, init='normal', activation='tanh'))
# Add dropout
model.add(Dropout(0.5))

# Add a fully connected layer
model.add(Dense(10, init='normal', activation='tanh'))

# Add a fully connected layer
model.add(Dense(1, init='normal'))

###################
# model training 
###################

model.compile(loss = 'mse', optimizer = 'adam') #rmsprop
#model.compile(loss = 'mse', optimizer = Adam(lr=0.0004))
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*3, validation_data = validation_generator, nb_val_samples = len(validation_samples)*3, nb_epoch = 10)

model.save('model.h5')
print("model saved")