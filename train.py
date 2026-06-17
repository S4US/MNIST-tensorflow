import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# x_train on pikseli data ja y on klassifikaatio
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisoidaan data lukuihin 0 ja 1 välillä
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Luodaan tyhjä neuroverokko malli
model = tf.keras.models.Sequential()

# Tämä muuttaa taulukon pikseleitä yhdeksi pitkäksi pötköksi
# esim [[1, 2, 3],[1, 2, 3]] = 1, 2, 3, 1, 2, 3.
# Varmaan huono esimerkki mutta auttaa minua muistamaan
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Dense on yksinkertainen layeri jonka neuronit on täysin kytketty muihin
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))

# Viimeinen taso joka luokittelee loppu tuloksen. softmax muuttaa numerot todennäköisyyksiksi
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Valmistellaan malli adam = miten malli oppii, loss = mittaa kuinka väärässä malli on,
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Opettaa mallin MNIST datalla
model.fit(x_train, y_train, epochs=5)

model.save('mnist.keras')