# Monet Style Image Translation with CycleGAN

## Project Overview

[CycleGAN](https://ar5iv.labs.arxiv.org/html/1703.10593) is an unpaired image-to-image translation framework that learns to map between two visual domains without requiring paired examples. In this project, we apply CycleGAN to transform real photographs into Monet-style paintings.

The model consists of two generators (one for Photo → Monet, and one for Monet → Photo) and two discriminators (one per domain). These networks are trained adversarially: each generator tries to produce outputs that the opposite discriminator cannot distinguish from real images, while a cycle-consistency loss ensures that translating from Photo → Monet → Photo (and vice versa) recovers the original image.

As a result, the trained CycleGAN learns to render natural photos with the soft brushstrokes and color palette characteristic of Monet’s impressionist style.  
➡️ [CycleGAN Paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/1703.10593)  
➡️ [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)

## Model Architecture

We implement the standard [CycleGAN architecture](https://ar5iv.labs.arxiv.org/html/1703.10593). Each generator network:
- Downsamples the input using convolutional layers
- Applies several ResNet blocks
- Upsamples back to the original image size

We use **Instance Normalization** instead of Batch Normalization for more stable training and style transfer.  
➡️ [Instance Norm in TensorFlow](https://www.tensorflow.org/tutorials/generative/cyclegan)

The discriminators are **PatchGANs** that look at 70×70 patches to determine real vs. fake images.

### Summary:
- **G**: Photo → Monet (Generator)
- **F**: Monet → Photo (Generator)
- **D<sub>Y</sub>**: Discriminator for Monet images
- **D<sub>X</sub>**: Discriminator for Photo images

This setup enforces:
- `G: X → Y`
- `F: Y → X`
- `D<sub>Y</sub>` distinguishes real vs generated Monet images
- `D<sub>X</sub>` distinguishes real vs generated photos

The architecture uses 9 ResNet blocks for 256×256 images, consistent with the original [CycleGAN paper](https://www.tensorflow.org/tutorials/generative/cyclegan).
