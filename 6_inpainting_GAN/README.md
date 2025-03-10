# Description
This code implements a Generative Adversarial Network (GAN) to perform inpainting on [overhead infrared (IR) images of a series of prescribed fires on an area of 10 x 10 m](https://www.fs.usda.gov/rds/archive/catalog/RDS-2022-0076). The GAN has two sub-models, the context encoder (generator) and patch discriminator (discriminator).

- `main.py` trains and saves the models. During each epoch, the results (ground truth, masked images and inpainted images) are saved to 'results'. **Does not employ train test split.** The training conditions are saved in a TXT file in 'models/{args.output_dir}', epoch number and losses are logged in a CSV file, plotted and saved in 'results'.
- `inference.py` loads a trained generator and performs inferencing, ideally on dataset it has never seen during its training phase.
- `models.py` contains the definition of the sub-models.
- `image_utils.py` contains functions needed to load, preprocess and perform masking on the dataset. Also contains function to plot losses against epoch from the CSV file.

# Usage - main.py
## Parameters
```
--data_dir          Base directory containing folders of images used to train the GAN
--folders           List of subfolder names to load dataset from
--img_scaled_dim    The dimension of the longest side of the image after scaling
--coverage          Fraction of image to be masked
--epochs            Number of epochs
--batch_size        The number of images in each mini-batch
--lambda_recon      Weight of reconstruction loss in generator's loss
--lr                Learning rate of Adam optimizer
--beta1             Beta1 (decay rate) of Adam optimizer
--output_dir        Name of folder to store the models and results
--num_show          Number of images to be displayed
```

**Example**
```
python main.py --data_dir dataset/IR_images --folders 3 5 6 7 --img_scaled_dim 320 --coverage 0.15 --epochs 50 --batch_size 32 --lambda_recon 10 --lr 0.0002 --beta1 0.5 --output_dir IR_images_3_5_6_7 --num_show 5
```

# Usage - inference.py
## Parameters
```
--data_dir          Base directory containing folder of images to perform inference on
--model_path        Directory to load trained generator from
--img_scaled_dim    The dimension of the longest side of the image after scaling
--coverage          Fraction of image to be masked
--batch_size        The number of images in each mini-batch
--output_dir        Name of folder to store inference results
--num_show          Number of images to be displayed
```

**Example**
```
python inference.py --data_dir dataset --model_path models/test2/generator.pth --img_scaled_dim 320 --coverage 0.15 --batch_size 5 --output_dir inference_results --num_show 5
```

# Results
test1 and test2 in 'results' contain the training conditions for currently known best-known parameters.