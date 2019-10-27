import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as k

data = pd.read_csv("train_foo.csv")
dataset = np.array(data)
np.random.shuffle(dataset)
x = dataset
y = dataset
x = x[:, 1:2501]
y = y[:, 0]
x_train = x[0:12000, :]
x_train = x_train/255.
x_test = x[12000:13201, :]
x_test = x_test/255.

y = y.reshape(y.shape[0], 1)
y_train = y[0:12000, :]
y_train = y_train.T
y_test = y[12000:13201, :]
y_test = y_test.T

print("number of training examples: "+ str(x_train.shape[0]))
print("number of test examples: "+ str(x_test.shape[0]))
print("x_train shape: "+ str(x_train.shape))
print("y_train shape: "+ str(y_train.shape))
print("x_test shape: "+ str(x_test.shape))
print("y_test shape: "+ str(y_test.shape))

image_x = 50
image_y = 50
train_y = np_utils.to_categorical(y_train)
test_y = np_utils.to_categorical(y_test)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)
print("x_train shape: "+ str(x_train.shape))
print("x_test shape: "+ str(x_test.shape))
print("y_train shape: "+str(train_y.shape))

def keras_model(image_x, image_y):
    num_of_classes = 12
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(5, 5), padding="same"))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emojinator.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint1]

    return model, callback_list

model, callbacks_list = keras_model(image_x, image_y)
model.fit(x_train, train_y, validation_data=(x_test, test_y), epochs=1 , batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(x_test, test_y, verbose=0)
print("CNN error: %.2f%%" % (100 - scores[1]*100))
print_summary(model)

model.save('emojinator.h5')




















