import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

featuresdf = pd.read_csv('features.csv')

x = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size = 0.19906432748, random_state = 42)
print(x.size)
num_rows = 40
num_columns = 428
num_channels = 1

#x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
#x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, input_shape = (num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32,kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64,kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128,kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

model.summary()

score = model.evaluate(x_test, y_test, verbose = 1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 72
batch_size = 256

checkpointer = ModelCheckpoint(filepath = '../saved_models/weights.best.basic_cnn.hdf5',
                                verbose = 1, save_best_only = True)

start = datetime.now()

model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs, validation_data = (x_test, y_test), callbacks = [checkpointer], verbose = 1)

duration = datetime.now() - start
print("Training completed in: ", duration)

score = model.evaluate(x_train, y_train, verbose = 0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose = 0)
print("Testing Accuracy: ", score[1])