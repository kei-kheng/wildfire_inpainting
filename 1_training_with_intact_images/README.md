# Description
This code attempts to train an inpainting/denoising autoencoder using 400 128x128 intact (unmasked) images of ellipses.
The model was trained and saved using main.py, where the model learns the latent space representation of the intact images.
The model was loaded and validated using inpainting.py, where masked inputs are fed to the model and the results are visualised.

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
--train_dir      Directory containining the images of ellipses used to train (and test) the autoencoder. Train: 90% of images, Test: 10% of images (This is visualised)
--epochs         Number of epochs 
--batch_size     The number of images in each mini-batch
--model_out      Directory to save trained autoencoder
```

**Example**
```
python main.py --train_dir dataset/training --epochs 5 --batch_size 8 --model_out model/autoencoder.pth
```

# Usage - inpainting.py
## Parameters (optional)
```
--data_dir       Directory containing the images used to validate the autoencoder (different from the images used to train the autoencoder)
--model_path     Directory to load the trained model from
--coverage       Fraction of pixels to set to 0 (black) in mask 
--num_show       Number of reconstructed images to display
```

**Example**
```
python inpainting.py --data_dir dataset/testing --model_path model/autoencoder.pth --coverage 0.3 --num_show 5
```

# Results
From the execution of main.py, it can be seen that the model could learn and reconstruct the original images from the latent space represention in the training phase.
However, when the trained model was fed with masked inputs, it could not reconstruct the images accurately.

## Cause
The model was trained using intact images. Zeroed out pixles were treated as valid inputs. A better way would be to train the model using masked images, where the autoencoder learns to reconstruct the missing pixels in the training phase.