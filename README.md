# Project Goal
Implement a generative model to perform image inpainting on lossy images of wildfire scenes.

Lossy image --> Denoising autoencoder --> Inpainting GAN --> Reconstructed, denoised image

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)
- [PartialConv2d](https://github.com/NVIDIA/partialconv)

# Project Progress
Sucessful implementation of single-agent exploration scenario for grayscale ellipses. However, it was realised that the current model architecture only performs denoising and not inpainting. The next steps are:
- [X] Revised model architecture to perform inpainting on grayscale images - **unsatisfactory results**. Tried CAE, PCAE and U-Net.
- [X] Multi-agent exploration scenario.
- [X] Dataset: Export individual frames from .seq files (provided by Zak) as JPG/PNG.
- [X] Revise model architecture to perform inpainting on RGB images - **GANs**.
---------------------------------------------------------------------------------------------------
- [ ] Train GAN on a larger dataset (~500 at the moment) for better generalization. Save one set of data for inferencing (not used in training).
- [ ] Extend main.py to write the epoch number and losses (2) to a CSV file (pandas).
- [ ] Extend main.py to write the training conditions to a TXT file.
- [ ] Implementation of RGB environment.
---------------------------------------------------------------------------------------------------
- [ ] Masking the training dataset with different types of noise:
    - [X] Random noise
    - [X] 'Square' noise
    - [X] 'Agents: Explored region' noise
    - [ ] Other types of noise
- [ ] Improvement of random walk policy
- [ ] Upload zipped folders of dataset to GitHub

**Check logbook for saved links that might be useful!**