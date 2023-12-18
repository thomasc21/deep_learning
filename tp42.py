import numpy as np
import tensorflow as tf
cifar = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = cifar.load_data()
  
x_train, x_test = x_train / 255.0 , x_test / 255.0
input_layer = tf.keras.layers.Input(name="input_layer", shape=(None, None,1))

# modifier le conv2D
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128)(input_layer)
#add 3 conv2D

class MyCustomLayer(tf.keras.layers.Layer):
 
 def __init__(self, custom_parameter=42, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_parameter = custom_parameter
 def build(self, inputs_shape):
    self.kernel = self.add_weight("kernel", shape=inputs_shape)
 def call(self, inputs):
    # Define the forward computation using inputs and self.kernel
    return inputs + self.kernel
 def get_config(self):
    config = super().get_config()
    config.update({
    # must have the same name as the self. method
    "custom_parameter": self.custom_parameter
    })
    return config



input_to_block = x
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)
for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block

input_to_block = x
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)
for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block

input_to_block = x
x = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=32 ,padding="same")(x)
for _ in range ((7-1)//2):
    x = tf.keras.layers.Conv2D(kernel_size=(1,3), filters=32 ,padding="same")(x)
    x = tf.keras.layers.Conv2D(kernel_size=(3,1), filters=32 ,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
x= tf.keras.layers.Conv2D(kernel_size=(1,1), filters=128 ,padding="same")(x)
x += input_to_block






hidden_layer = tf.keras.layers.Conv2D(kernel_size=(1,1), filters=10, padding="same")(x)
hidden_layer = tf.keras.layers.GlobalAveragePooling2D()(hidden_layer)
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(hidden_layer)
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
model.summary(100)
model.compile(
    optimizer = "Adam",
    loss = {
        "output_layer": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    metrics = ["acc"],
)
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size = 32,
    epochs = 5
)