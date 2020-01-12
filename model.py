import csv
import cv2
import numpy as np

# lines = []
images = []
# measurements = []
angles = []

def processImage(IMG):

    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG = cv2.resize(IMG, (320, 160), interpolation = cv2.INTER_AREA)

    return IMG

samples = []

# with open('../newdata1/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../newdata1/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

# with open('../data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../data/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

# with open('../simdata1/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../simdata1/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

# with open('../simdata2/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../simdata2/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

# with open('../simdata3/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../simdata3/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

# with open('../turningdata/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../turningdata/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1


# with open('../turningdata2/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../turningdata2/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1


# with open('../turningdata3/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     counter = 0

#     for line in reader:
#         if counter != 0:
#             for i in range(0,3):
#                 source_path = line[i]
#                 filename = source_path.split('/')[-1]
#                 # for windows path conventions
#                 filename = filename.split('\\')[-1]
#                 current_path = '../turningdata3/IMG/' + filename
#                 line[i] = current_path
#             samples.append(line)
#         counter += 1

with open('../edgeside3laps/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0

    for line in reader:
        if counter != 0:
            for i in range(0,3):
                source_path = line[i]
                filename = source_path.split('/')[-1]
                # for windows path conventions
                filename = filename.split('\\')[-1]
                current_path = '../edgeside3laps/IMG/' + filename
                line[i] = current_path
            samples.append(line)
        counter += 1

with open('../edgesideBackwards3laps/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0

    for line in reader:
        if counter != 0:
            for i in range(0,3):
                source_path = line[i]
                filename = source_path.split('/')[-1]
                # for windows path conventions
                filename = filename.split('\\')[-1]
                current_path = '../edgesideBackwards3laps/IMG/' + filename
                line[i] = current_path
            samples.append(line)
        counter += 1

print("total sample numbers:", len(samples))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_image = processImage(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)


                # append flip images
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle*-1.0)

                # using side cameras
                correction = 0.2
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                file_path_left = batch_sample[1]
                image_left = cv2.imread(file_path_left)
                image_left = processImage(image_left)
                images.append(image_left)
                angles.append(steering_left)

                file_path_right = batch_sample[2]
                image_right = cv2.imread(file_path_right)
                image_right = processImage(image_right)
                images.append(image_right)
                angles.append(steering_right)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(46,(5,5),subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,(5,5),subsample=(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            steps_per_epoch=len(train_samples)/batch_size,
            validation_data=validation_generator,
            validation_steps=len(validation_samples)/batch_size,
            epochs=8, verbose=1)

model.save('model.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

plt.savefig('foo.png')
plt.savefig('foo.pdf')

# plt.show()



# oldModel = load_model('model_50dropout_ds13.h5')
# oldModel.fit_generator(train_generator,
#             steps_per_epoch=len(train_samples)/batch_size,
#             validation_data=validation_generator,
#             validation_steps=len(validation_samples)/batch_size,
#             epochs=5, verbose=1)


# oldModel.save('model_50dropout_dsnewonly.h5')



