# Project Goal
Implement a generative model to perform image inpainting on lossy images of wildfire scenes

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)

# Project Progress
Sucessful implementation of single-agent exploration scenario for grayscale ellipses. However, it was realised that the current model architecture only performs denoising and not inpainting. The next steps are:
- [x] Environment and agent, single-agent exploration scenario
- [] Revise model architecture to perform inpainting
- [] Different types of noise - take inspiration from latest version of James' code
- [] Training using Zak's images
- [] Multi-agent exploration scenario