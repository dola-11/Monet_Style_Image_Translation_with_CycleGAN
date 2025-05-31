import tensorflow as tf

class CycleGANLosses:
    def __init__(self):
        pass

    def discriminator_loss(self,real, generated):
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real), real, from_logits=True)
        generated_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(generated), generated, from_logits=True)
        return tf.reduce_mean(real_loss + generated_loss)

    def generator_loss(self,generated):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(generated), generated, from_logits=True))

    def cycle_loss(self,real_image, cycled_image):
        return tf.reduce_mean(tf.abs(real_image - cycled_image)) * 10.0

    def identity_loss(self,real_image, same_image):
        return tf.reduce_mean(tf.abs(real_image - same_image)) * 5.0
