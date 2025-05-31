from Model_Training.models import Models
from Model_Training.train import CycleGAN
from Model_Training.losses import CycleGANLosses
from Model_Training.data_loader import load_dataset
from Model_Training.config import MONET_DIR, PHOTO_DIR, EPOCHS
import os
import tensorflow as tf

photo_paths = [os.path.join(PHOTO_DIR, f) for f in os.listdir(PHOTO_DIR) if f.endswith(".jpg")]
monet_paths = [os.path.join(MONET_DIR, f) for f in os.listdir(MONET_DIR) if f.endswith(".jpg")]

photo_dataset = load_dataset(photo_paths)
monet_dataset = load_dataset(monet_paths)
train_dataset = tf.data.Dataset.zip((photo_dataset, monet_dataset))



cyclegan_model = CycleGAN(
    gen_G=Models.Generator(),
    gen_F=Models.Generator(),
    disc_X=Models.Discriminator(),
    disc_Y=Models.Discriminator()
)

cyclegan_model.compile(
    gen_G_opt=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    gen_F_opt=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    disc_X_opt=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    disc_Y_opt=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    gen_loss_fn=CycleGANLosses.generator_loss,
    disc_loss_fn=CycleGANLosses.discriminator_loss,
    cycle_loss_fn=CycleGANLosses.cycle_loss,
    identity_loss_fn=CycleGANLosses.identity_loss,
)

cyclegan_model.fit(train_dataset, epochs=EPOCHS)
