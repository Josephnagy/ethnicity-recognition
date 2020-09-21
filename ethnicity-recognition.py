import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
#import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import array_to_img

class Face:
    def __init__(self, filename, age, gender, race, date, time, img):
        self.filename = filename
        self.age = age
        self.gender = gender
        self.race = race
        self.date = date
        self.time = time
        self.img = img
#initialize variables
test_label = []
test_img = []
validation_label = []
validation_img = []
train_label = []
train_img = []

#def prepare_test(testArray, validArray, trainArray):
def prepare_test(testArray):

    i=0
    #test data
    while i<len(testArray):
        test_label.append(testArray[i].race)
        test_img.append(testArray[i].img)

    #reset i
    i=0
    #return tuple of arrays
    return (test_label, test_img)

def prepare_valid(validArray):
    i=0
    #validation data
    while i<len(validArray):
        valid_label.append(trainArray[i].race)
        valid_img.append(trainArray[i].img)

    #reset i
    i=0
    #return tuple of arrays
    return (valid_label, valid_img)

def prepare_train(trainArray):
    i=0
    #training data
    while i<len(trainArray):
        train_label.append(trainArray[i].race)
        train_img.append(trainArray[i].img)

    #reset i
    i=0
    #return tuple of arrays
    return (train_label, train_img)


faces = []
directory = '/Users/josephnagy/UTK_Face_files/test_faces'
os.chdir(directory)

#read in images and store them
#Filename format: [age]_[gender]_[race]_[date&time].jpg | d&t: yyyymmddHHMMSSFFF
#images elements: ([filename], [age], [gender], [race], [date], [time], [img])
for filename in os.listdir(directory):
    img = img_to_array(Image.open(filename))
    split_filename = filename.split("_")
    age = int(split_filename[0])
    gender = split_filename[1]
    race = split_filename[2]
    date = int(split_filename[3][:8])
    time = split_filename[3][8:].split(".")[0]
    fce = Face(filename, age, gender, race, date, time, img)
    faces.append(fce)

#split data into training/testing  (80/20 split)
#shuffled_faces = np.random.shuffle(faces)
#testing_faces = shuffled_faces[0:int(len(shuffled_faces))*0.2]
#training_faces= shuffled_faces[int(len(shuffled_faces))*0.20:]
np.random.shuffle(faces)
length = int(len(faces)*0.2)

testing_faces = faces[0:length]
training_faces= faces[length:]


#split training data into training/validation (80/20 split again)
validation_faces = faces[0:length]
training_faces= faces[length:]

#organize data into parallel lists
test_label = prepare_test(testing_faces)[0]
test_img = prepare_test(testing_faces)[1]
validation_label = prepare_valid(validation_faces)[0]
validation_img = prepare_test(validation_faces)[1]
train_label = prepare_test(training_faces)[0]
train_img = prepare_test(training_faces)[1]


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]
             )

model.fit(train_img,
          train_label,
          epochs = 8,
         # validation_data=validation_img, validation_label,
         # steps_per_epoch=2,
         # validation_steps=9
         )

test_acc, test_loss = model.evaluate(test_img, test_label)
print("Accuracy:", test_acc)
