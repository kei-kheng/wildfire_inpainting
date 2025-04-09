# Description
Trains and compares 3 different inpainting models:
- Convolutional Autoencoder (CAE)
- Partial Convolutional Autoencoder (PCAE)
- Generative Adrversarial Network (GAN)

**Evaluation metrics:** MSE, PSNR, SSIM. Calculated over **inpainted portions** of image.

# Usage - main.py
## Parameters
```
--yaml_path    Path of config.yaml (if existent)
--data_dir     Directory to load training images from
--model_type   Specifies type of model to train, takes values within set ["CAE", "PCAE", "GAN"]
--output_dir   Directory to store results
--coverage     Size of mask relative to the size of image
--epochs       Number of epochs
--batch_size   Batch size
```

**Example**
```
python main.py --data_dir dataset --model_type CAE --output_dir test --coverage 0.15 --epochs 50 --batch_size 16
```