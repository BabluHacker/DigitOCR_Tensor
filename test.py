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
from keras.models import load_model
model=load_model('/home/mehedi/AI/MAIN/test/ImgClass.model')
'''
img=load_img('E4_22.jpg')
img=img_to_array(img).astype('float32')
img=img.reshape((1,)+img.shape)
img=img/255.0

print "loaded"

pr = model.predict_classes(img, 32, 1)
print str(pr[0])


pr=model.predict(img,32,1)
print pr
print "printing last"
print model.evaluate_generator
'''