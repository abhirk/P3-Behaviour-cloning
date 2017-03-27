import csv
import cv2
import os
import traceback
import numpy as np
import tqdm
from sklearn.utils import shuffle

CSV_FILE = "data/driving_log.csv"
IMG_DIR = "data/IMG/"
STEERING_CORRECTION = 0.2

images = []
measurements = []

samples = []
with open(CSV_FILE) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = IMG_DIR + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Collecting dataset...."),
try:
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
except:
    traceback.print_exc()
print("done")


from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
#from keras.layers import Conv2D, MaxPooling2D

print("Training model...")
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3),  output_shape=(160,320,3)))
#cropping default 50,20
model.add(BatchNormalization())
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten()) #Flatten(input_shape=(160,320,3)
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)
model.save("model.h5")

