# Project Goal
Implement a generative model to perform image inpainting on lossy images of wildfire scenes

Lossy image --> Denoising autoencoder --> Inpainting GAN --> Reconstructed, denoised image

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)
- [PartialConv2d](https://github.com/NVIDIA/partialconv)

# Project Progress
Sucessful implementation of single-agent exploration scenario for grayscale ellipses. However, it was realised that the current model architecture only performs denoising and not inpainting. The next steps are:
- [X] Revised model architecture to perform inpainting on grayscale images - **unsatisfactory results**. Tried:
    - [X] U-Net
    - [X] Partial convolution: https://arxiv.org/abs/1804.07723, https://github.com/NVIDIA/partialconv
- [X] Decoupled encoder and decoder
- [X] Multi-agent exploration scenario
- [ ] Dataset: Export individual frames from .seq files (provided by Zak) as JPG/PNG
- [ ] Revise model architecture to perform inpainting on RGB images - [**GANs**](https://medium.com/towards-data-science/inpainting-with-ai-get-back-your-images-pytorch-a68f689128e5)
- [ ] Masking the training dataset with different types of noise
    - [X] Random noise
    - [X] 'Square' noise
    - [X] 'Agents: Explored region' noise
    - [ ] Other types of noise
- [ ] Improvement of random walk policy

**Check logbook for saved links that might be useful!**