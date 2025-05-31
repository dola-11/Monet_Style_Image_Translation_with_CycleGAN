import os
import tensorflow as tf
from PIL import Image
import numpy as np
from config import HEIGHT, WIDTH, BATCH_SIZE, BUFFER_SIZE

class ImageDataset:
    def __init__():
       pass

    def decode_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [HEIGHT, WIDTH])
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        return image

    def load_dataset(image_paths, training=True):
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.shuffle(BUFFER_SIZE)
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    
if __name__ == "__main__":
    load_image = ImageDataset()


