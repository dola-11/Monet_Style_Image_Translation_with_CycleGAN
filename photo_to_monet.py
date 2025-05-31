from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


gen_G = load_model('Models/gen_G (1).keras', compile=False)


def preprocess_image(path):
    img = Image.open(path).resize((256, 256))  # or (512, 512) if trained that way
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

input_image = preprocess_image('landscape.jpg')


monet_image = gen_G(input_image, training=False)


monet_image = (monet_image[0] + 1.0) / 2.0  
monet_image_uint8 = (monet_image.numpy() * 255).astype(np.uint8)
Image.fromarray(monet_image_uint8).save("monet_output.jpg")


plt.imshow(monet_image)
plt.axis('off')
plt.title("Generated Monet Image")
plt.show()
