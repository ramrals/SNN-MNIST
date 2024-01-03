import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Descargar y cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir los datos de salida a one-hot encoding
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

class SNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Capa de entrada
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(28, 28))

        # Capa LSTM
        self.lstm_layer = tf.keras.layers.LSTM(128, activation='leaky_relu')

        # Capa de salida
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.lstm_layer(x)
        x = self.output_layer(x)
        return x

# Definir el optimizador y la función de pérdida
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compilar el modelo
model = SNN()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Entrenamiento
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Guardar el modelo
model.save('mi_modelo.h5')
