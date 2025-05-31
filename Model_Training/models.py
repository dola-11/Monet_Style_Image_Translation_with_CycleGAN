import tensorflow as tf
from keras import layers as L, Model
from layers import layers
from config import HEIGHT, WIDTH, CHANNELS


class Models:
    def __init__(self):
        pass

    def Generator(self):
        inputs = L.Input(shape=[HEIGHT, WIDTH, CHANNELS])
        down_stack = [
            layers.downsample(64, 4, apply_norm=False),
            layers.downsample(128, 4),
            layers.downsample(256, 4),
        ]
        up_stack = [
            layers.upsample(128, 4),
            layers.upsample(64, 4),
        ]
        x = inputs
        for down in down_stack:
            x = down(x)
        for up in up_stack:
            x = up(x)
        last = L.Conv2DTranspose(CHANNELS, 4, strides=2, padding='same',
                                kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                activation='tanh')
        x = last(x)
        return Model(inputs=inputs, outputs=x)

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = L.Input(shape=[HEIGHT, WIDTH, CHANNELS])
        x = layers.downsample(64, 4, False)(inp)
        x = layers.downsample(128, 4)(x)
        x = layers.downsample(256, 4)(x)
        last = L.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)
        return Model(inputs=inp, outputs=last)
