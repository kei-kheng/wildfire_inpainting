# Description
This code implements a Generative Adversarial Network (GAN) to perform inpainting on [overhead infrared (IR) images of a series of prescribed fires on an area of 10 x 10 m](https://www.fs.usda.gov/rds/archive/catalog/RDS-2022-0076). The GAN has two sub-models, the context encoder (generator) and patch discriminator (discriminator).

- `main.py` trains and saves the models. During each epoch, the results (ground truth, masked images and inpainted images) are saved to 'results'. **Does not employ train test split.** The training conditions are saved in a TXT file in 'models/{args.output_dir}', epoch number and losses are logged in a CSV file in 'results'.
- `inference.py` loads a trained generator and performs inferencing, ideally on dataset it has never seen during its training phase.
- `models.py` contains the definition of the sub-models.
- `image_utils.py` contains functions needed to load, preprocess and perform masking on the dataset.

# Usage - main.py
## Parameters
```
--data_dir         Directory containing images use to train the GAN
--img_scaled_dim   The dimension of the longest side of the image after scaling
--coverage         Fraction of image to be masked
--epochs           Number of epochs
--batch_size       The number of images in each mini-batch
--lambda_recon     Weight of reconstruction loss in generator's loss
--lr               Learning rate of Adam optimizer
--beta1            Beta1 (decay rate) of Adam optimizer
--output_dir       Name of folder to store the models and results
--num_show         Number of images to be displayed
```

**Example**
```
python main.py --data_dir dataset/synthetic_coloured_ellipses --img_scaled_dim 80 --coverage 0.15 --epochs 50 --batch_size 64 --lambda_recon 10 --lr 0.0002 --beta1 0.5 --output_dir synthetic_coloured_ellipses --num_show 5
```