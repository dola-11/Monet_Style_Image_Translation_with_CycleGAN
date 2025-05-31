import tensorflow as tf
from losses import *
from config import EPOCHS
from data_loader import load_dataset

class CycleGAN(tf.keras.Model):
    def __init__(self, gen_G, gen_F, disc_X, disc_Y):
        super().__init__()
        self.gen_G = gen_G
        self.gen_F = gen_F
        self.disc_X = disc_X
        self.disc_Y = disc_Y

    def compile(self, gen_G_opt, gen_F_opt, disc_X_opt, disc_Y_opt,
                gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super().compile()
        self.gen_G_opt = gen_G_opt
        self.gen_F_opt = gen_F_opt
        self.disc_X_opt = disc_X_opt
        self.disc_Y_opt = disc_Y_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)
            cycled_x = self.gen_F(fake_y, training=True)
            cycled_y = self.gen_G(fake_x, training=True)
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            disc_real_x = self.disc_X(real_x, training=True)
            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            gen_G_loss = self.gen_loss_fn(disc_fake_y)
            gen_F_loss = self.gen_loss_fn(disc_fake_x)
            cycle_loss_val = self.cycle_loss_fn(real_x, cycled_x) + self.cycle_loss_fn(real_y, cycled_y)
            total_gen_G_loss = gen_G_loss + cycle_loss_val + self.identity_loss_fn(real_y, same_y)
            total_gen_F_loss = gen_F_loss + cycle_loss_val + self.identity_loss_fn(real_x, same_x)
            disc_X_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        self.gen_G_opt.apply_gradients(zip(tape.gradient(total_gen_G_loss, self.gen_G.trainable_variables), self.gen_G.trainable_variables))
        self.gen_F_opt.apply_gradients(zip(tape.gradient(total_gen_F_loss, self.gen_F.trainable_variables), self.gen_F.trainable_variables))
        self.disc_X_opt.apply_gradients(zip(tape.gradient(disc_X_loss, self.disc_X.trainable_variables), self.disc_X.trainable_variables))
        self.disc_Y_opt.apply_gradients(zip(tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables), self.disc_Y.trainable_variables))

        return {
            "gen_G_loss": total_gen_G_loss,
            "gen_F_loss": total_gen_F_loss,
            "disc_X_loss": disc_X_loss,
            "disc_Y_loss": disc_Y_loss,
        }
