import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageProcessor:
    def __init__(self):
        pass

    def preprocess_image(path):
        img = Image.open(path).resize((256, 256))
        img = np.array(img).astype(np.float32)
        img = (img / 127.5) - 1.0
        return np.expand_dims(img, axis=0)

    def display_image(image):
        image = (image[0] + 1.0) / 2.0
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def save_image(tensor, path):
        img = (tensor.numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
