# Project Goal
- Implement a generative model to perform image inpainting on lossy images of wildfire scenes.
- Investigate collective perception in multi-agent systems.

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)
- [PartialConv2d](https://github.com/NVIDIA/partialconv)
- [Dataset of Overhead IR Images of 10x10m Fire](https://www.fs.usda.gov/rds/archive/catalog/RDS-2022-0076)

# Future Work
- Improve model to generalize better: modify architecture to use **partial convolution**, introduce **different masks/noises**.
- Compressed representation of images, e.g., train an autoencoder to reduce communication overhead between agents. For example:
  - **On drone**: The noisy image is first passed through a (denoising) autoencoder's encoder to be (denoised and) compressed into latent representation.
  - **On central server**: Server receives agents' observation, passes them through autoencoder's decoder and finally an inpainting GAN.
