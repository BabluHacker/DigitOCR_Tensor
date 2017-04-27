import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras
import h5py

paths = ['/home/mehedi/AI/MAIN/test/Dataset']
x_train = []
y_train = []
n_class = 20
batch_size = 32
nb_epochs = 20
classes = {
	'E0': 0,
	'E1': 1,
	'E2': 2,
	'E3': 3,
	'E4': 4,
	'E5': 5,
	'E6': 6,
	'E7': 7,
	'E8': 8,
	'E9': 9,
	'B0': 10,
	'B1': 11,
	'B2': 12,
	'B3': 13,
	'B4': 14,
	'B5': 15,
	'B6': 16,
	'B7': 17,
	'B8': 18,
	'B9': 19,
}


def load_data_set():
    i=0
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:

                #print filename
                fullpath = os.path.join(root, filename)

                img = load_img(fullpath)

                img = img_to_array(img)
                #print fullpath+" "+str(img.shape)
                if img.shape[0] !=100 or img.shape[1]!=100 or img.shape[2]!=3:
                    print fullpath +" "+str(img.shape)
                x_train.append(img)
                t = fullpath.rindex('/')
                #i=i+1
                y_train.append(classes[fullpath[t+1:t+3]])





def make_network(x_train):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(n_class))
    model.add(Activation('softmax'))

    return model

def train_model(model, X_train, Y_train):
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epochs)

load_data_set()
a = numpy.asarray(y_train)
#print y_train
y_train_new = a.reshape(a.shape[0], 1)
Y_train = np_utils.to_categorical(y_train_new, n_class)
x_train=numpy.array(x_train).astype('float32')
x_train=x_train/255.0
#print x_train.shape
model=make_network(numpy.array(x_train))
train_model(model,x_train,Y_train)
model.save('/home/mehedi/AI/MAIN/test/ImgClass.model')



