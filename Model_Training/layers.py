import tensorflow as tf
from keras import layers as L

Norm = lambda: tf.keras.layers.LayerNormalization(epsilon=1e-5)

class layers:

    def __init__(self):
        pass

    def downsample(self,filters, size, apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(L.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))
        if apply_norm:
            result.add(Norm())
        result.add(L.LeakyReLU())
        return result

    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(L.Conv2DTranspose(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))
        result.add(Norm())
        if apply_dropout:
            result.add(L.Dropout(0.5))
        result.add(L.ReLU())
        return result
