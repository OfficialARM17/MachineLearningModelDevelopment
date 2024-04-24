# import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
# make use fo sklearn to test and train the dataset
from sklearn.model_selection import train_test_split
from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential, Model, load_model
# define the dataset that will be used
path="SignLanguageDataset"
# go to this location
files=os.listdir(path)
# sort files from A-Y
files.sort()
# print to see list for debugging
print(files)
# create lists of images and labels
image_array = []
label_array = []
# loop through each file in files
for i in tqdm(range(len(files))):
	# list of every image in every folder
	sub_file=os.listdir(path+"/"+files[i])
	# loop through each sub folder
	for j in range(len(sub_file)):
		# path of each image
		#Example:SignLanguageDataset/A/image_name1.jpg
		file_path=path+"/"+files[i]+"/"+sub_file[j]
		# read each image
		image=cv2.imread(file_path)
		# resize image by 96x96 (needed for the Android Application)
		image=cv2.resize(image,(72,72))
		# convert BGR image to RGB image (needed for the Android Application)
		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# add this image to image_array
		image_array.append(image)
		# add label to label_array
		label_array.append(i)
# convert the lists to arrays
image_array=np.array(image_array)
label_array=np.array(label_array,dtype="float")

# split the dataset into testing and training
# split between training and testing will be 85:15
X_train,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)
del image_array,label_array
# needed to free memory 
import gc
gc.collect()
# Create a Sequential model
model=Sequential()
# add pretrained models to Sequential model
pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(72,72,3),include_top=False)
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.build(input_shape=(None,72,72,3))

# Compile the model
model.compile(optimizer="adam",loss="mae",metrics=["mae"])

# Create callbacks for model checkpoint and reducing the learning rate
ckp_path="sign_language_model/model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    monitor="val_mae",
    mode="auto",
    save_best_only=True,
    save_weights_only=True)
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.9,
    monitor="val_mae",
    mode="auto",
    cooldown=0,
    patience=5,
    verbose=1,
    min_lr=1e-6)
# Start training model

Epochs=100
Batch_Size=32
history=model.fit(
    X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    batch_size=Batch_Size,
    epochs=Epochs,
    callbacks=[model_checkpoint,reduce_lr])
# Load the best model weights
model.load_weights(ckp_path)

# convert model to TFLite model
# required to use in Android Studio
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

# save model
with open("sign_language_model.tflite","wb") as f:
	f.write(tflite_model)

# Make predictions based on the dataset
prediction_val=model.predict(X_test,batch_size=32)

# print the first 5 prediciton values and their corresponding labels
print(prediction_val[:5])
print(Y_test[:5])
