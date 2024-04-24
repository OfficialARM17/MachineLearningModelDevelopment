# Import required libraries
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the dataset path
path = "HandDataset"
categories = os.listdir(path)
categories.sort()

# Load images and labels
image_array = []
label_array = []
for i, category in enumerate(categories):
    images = os.listdir(os.path.join(path, category))
    for image_name in images:
        image_path = os.path.join(path, category, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array.append(image)
        label_array.append(i)

# Convert lists to arrays
image_array = np.array(image_array)
label_array = np.array(label_array)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.2)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images for CNN input
X_train = X_train.reshape(-1, 100, 100, 1)
X_test = X_test.reshape(-1, 100, 100, 1)

# Create CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test Accuracy:", test_accuracy)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model as "hand_model.tflite"
with open("hand_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model saved as hand_model.tflite")
