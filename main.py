
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import numpy as np


# Import the dataset
mnist = datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define network parameters
n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_hidden4 = 64
n_hidden5 = 32
n_hidden6 = 16
n_output = 10
learning_rate = 1e-4
n_iterations = 15
batch_size = 128
dropout = 0

# Create model
model = models.Sequential()
model.add(layers.Dense(n_hidden1, activation='relu', input_shape=(n_input,)))
model.add(layers.Dense(n_hidden2, activation='relu'))
model.add(layers.Dense(n_hidden3, activation='relu'))
model.add(layers.Dense(n_hidden4, activation='relu'))
model.add(layers.Dense(n_hidden5, activation='relu'))
model.add(layers.Dense(n_hidden6, activation='relu'))
model.add(layers.Dropout(dropout))
model.add(layers.Dense(n_output, activation='softmax'))


# Define loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=n_iterations, batch_size = batch_size)

img_path = "/home/med/Desktop/MASTER_STRI/S3/IA/tensorflow-demo/numeros/7.jpeg"
img = load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_array = img_to_array(img)
img_array1 = img_array.reshape(1, 784)
img_array1 /= 255.0
predictions = model.predict(img_array1)
predicted_digit = np.argmax(predictions[0])
print(predictions)
print('Predicted digit:', predicted_digit)

cv2.imshow('Image originale', img_array)
cv2.waitKey(0)

