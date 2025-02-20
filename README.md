# Project Goal
Implement a generative model to perform image inpainting on lossy images of wildfire scenes

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)
- [PartialConv2d](https://github.com/NVIDIA/partialconv)

# Project Progress
Sucessful implementation of single-agent exploration scenario for grayscale ellipses. However, it was realised that the current model architecture only performs denoising and not inpainting. The next steps are:
- [X] Revise model architecture to perform inpainting on grayscale images - **unsatisfactory results**
    - Options: 
        - [X] U-Net
        - [X] Partial convolution: https://arxiv.org/abs/1804.07723, https://github.com/NVIDIA/partialconv
        - [ ] Gated convolution: https://arxiv.org/abs/1806.03589, https://github.com/JiahuiYu/generative_inpainting
- [X] Decouple encoder and decoder (might be unnecessary)
- [ ] Dataset: Export individual frames from .seq files (provided by Zak) as JPG/PNG
- [ ] Perform inpainting on RGB images
- [ ] Multi-agent exploration scenario, e.g., additional arguments in Agent's constructor: `shared_observed, shared_explored`
- [ ] Masking the training dataset with different types of noise
    - [ ] Inspiration from James' code
    - [ ] Other types of noise - **implemented 'square' noise**
- [ ] Improvement of random walk policy

**Check logbook for saved links that might be useful!**