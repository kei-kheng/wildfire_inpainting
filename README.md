# Project Goal
- Implement a generative model to perform image inpainting on lossy images of wildfire scenes.
- Investigate collective perception in multi-agent systems.

Lossy image --> Denoising autoencoder --> Inpainting GAN --> Reconstructed, denoised image

# Links to Resources Used
- [James' Autoencoder](https://github.com/JamesHarcourt7/autoencoder-perception)
- [2D Shape Generator](https://github.com/TimoFlesch/2D-Shape-Generator)
- [PartialConv2d](https://github.com/NVIDIA/partialconv)

# To-do [Logbook - 10 March 2025]
*Model & Dataset*
- [ ] Improve model to generalize better: expand dataset, train for more epochs.
- [X] Data augmentation: rotation, random number of square masks with varying sizes.

*Environment & Agent*
- [X] RGB environment
- [ ] List down assumptions: agents' observation patch, information decay rate and **communication** range.

*Performance*
- [ ] Evaluation metrics, e.g., SSIM, PNSR.
- [ ] Investigate adaptability of model in a dynamic environment.

- [ ] Upload zipped folders of dataset to GitHub

**Check logbook for saved links that might be useful!**