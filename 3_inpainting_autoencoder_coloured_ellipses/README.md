# Description
This code attempts to train an inpainting/denoising autoencoder using 400 128x128 masked images of ellipses.

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
python main.py --data_dir dataset/training --epochs 5 --coverage 0.3 --batch_size 8 --model_out model/inpainting_autoencoder.pth --num_show 5
```

# Results
The model was trained using masked inputs. From the execution of main.py, it could be seen that this inpainting autoencoder could reconstruct the missing pieces better than the autoencoder which was trained using intact images.