# Description
An an extension of the previous convolutional autoencoder (CAE), this code attempts to perform image inpainting on colour-coded (function of fire intensity), 720x720 ellipses. The dataset was generated using `main_batch.py` in coloured_ellipses_generator.

# Setting Up
**Anaconda:**
```
conda env create -f environment.yml
```

**pip:**
```
pip install -r requirements.txt
```

# Usage - main.py
## Parameters (optional)
```
--data_dir      Directory containining the images of ellipses used to train (and validate) the autoencoder. Train: 90% of images, Test: 10% of images (This is visualised)
--epochs         Number of epochs
--coverage       Fraction of pixels to set to 0 (black) in mask 
--batch_size     The number of images in each mini-batch
--model_out      Directory to save trained autoencoder
--num_show       Number of reconstructed images to display
```

**Example**
```
python main.py --data_dir dataset_1000_images/training --epochs 5 --coverage 0.3 --batch_size 8 --model_out model/inpainting_autoencoder.pth  --num_show 10
```

# Results
For low number of epochs e.g., 1, 2, the model was only able to reconstruct the shape accurately, but not the colour. Accurate reconstruction was achieved by training the model on 1000 images with 30% noise over 5 epochs.

# Possible improvements to model architecture
- Use U-Nets (skip connection)
- Switch to GAN-based (Generative Adversarial Networks) methods