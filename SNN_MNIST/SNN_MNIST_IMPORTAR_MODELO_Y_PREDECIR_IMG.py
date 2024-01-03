# Importar las bibliotecas necesarias
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Descargar y cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cargar el modelo en formato TensorFlow SavedModel
model = tf.keras.models.load_model('mi_modelo')

# Hacer predicciones con el modelo cargado
predictions = model.predict(x_test)

# Obtener los índices de las clases predichas
predicted_classes = np.argmax(predictions, axis=1)

# Comparar los índices de las clases predichas con las etiquetas reales
correct_predictions = np.equal(predicted_classes, np.argmax(y_test, axis=1))

# Calcular la precisión
accuracy = np.mean(correct_predictions)

# Imprimir la precisión
print("Precisión después de cargar el modelo:", accuracy)
