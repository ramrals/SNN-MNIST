# Importar las bibliotecas necesarias
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Descargar y cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir los datos de salida a one-hot encoding
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# Definir la arquitectura de la SNN
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

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))


# Guardar el modelo en formato TensorFlow SavedModel
model.save('mi_modelo')

# Evaluar el modelo
model.evaluate(x_test, y_test)

# Obtener 10 índices aleatorios del conjunto de datos de prueba
random_indices = np.random.choice(x_test.shape[0], 10)

# Obtener las 10 imágenes correspondientes a los índices aleatorios
random_images = x_test[random_indices]

# Obtener las predicciones para las 10 imágenes
predictions = model.predict(random_images)

# Obtener los índices de las clases predichas
predicted_classes = np.argmax(predictions, axis=1)

# Mostrar las imágenes y las predicciones
for i in range(10):
    plt.imshow(random_images[i].reshape(28, 28))
    plt.title(f"Predicción: {predicted_classes[i]}")
    plt.show()



# Obtener los índices de las clases predichas
predicted_classes = np.argmax(predictions, axis=1)

# Comparar los índices de las clases predichas con las etiquetas reales
correct_predictions = np.equal(predicted_classes, np.argmax(y_test, axis=1))

# Calcular la precisión
accuracy = np.mean(correct_predictions)

# Imprimir la precisión
print("Precisión después de cargar el modelo:", accuracy)