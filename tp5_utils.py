import tensorflow as tf


"""Helper custom layers"""


class ResidualConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape
        self.layers = []
        for _ in range((self.kernel_size - 1) // 2):
            self.layers.append(
            tf.keras.layers.Conv2D(filters=c, kernel_size=(3, 1), padding="same")
            )
            self.layers.append(
            tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 3), padding="same")
            )
            self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.Activation("relu"))
            self.layers.append(tf.keras.layers.Dropout(0.2))


    def call(self, inputs):
        return inputs


class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self,kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size

    def get_config(self):
        config = super().get_config()
        config.update({})
        uptade = {"kernel_size":self.kernel_size}
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape
        self.jacki = ResidualConv2D(self.kernel_size)
        self.michel = tf.keras.layers.Conv2D(strides=2,kernel_size = self.kernel_size, filters = (2*c), padding="same")


    def call(self, inputs):
        x = self.jacki(inputs)
        y = self.michel(x)
        return x,y


class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self,kernel_size ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size


    def get_config(self):
        config = super().get_config()
        config.update({})
        uptade = {"kernel_size":self.kernel_size}
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape[0]
        self.paul =tf.keras.layers.Conv2DTranspose(filters=(c//2),kernel_size = self.kernel_size, padding = "same", strides=2)
        self.john = tf.keras.layers.Concatenate()
        self.marine = tf.keras.layers.Conv2D(filters=c//2, kernel_size=(1, 1), padding="same")
        self.george = ResidualConv2D(self.kernel_size)

    def call(self, inputs):
        self.paul(inputs[0])
        self.john([self.paul, inputs[1]])
        self.marine(self.john)
        return self.george(self.marine)
