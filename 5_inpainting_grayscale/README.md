# Description
This code attempts 3 types of architecture to perform inpainting on images of grayscale ellipses.

# Architecture and Results
- **Convolutional Autoencoder:** Poorly inpainted photos with checkerboard patterns. 'coverage' might be too large, ellipses might be too small.
- **Partial Convolutional Autoencoder:** Better inpainting capabilities than CAE but inpainted areas look like smeared ink.
- **U-Net:** Does not inpaint, might be due to 'wrong' architecture.

*The code snippet (class) of each model was stored in the `code_snippet_models` folder. PCAE has the best performance among the three.*

# Usage - main.py
## Parameters (optional)
```
--data_dir         Directory containing images use to train the model
--coverage         Fraction of image to be masked
--epochs           Number of epochs
--batch_size       The number of images in each mini-batch
--model_out        Directory to save trained model
--num_show         Number of images to be displayed
```

**Example**
```
python main.py --data_dir dataset/training --epochs 5 --batch_size 8 --model_out model/autoencoder.pth
```