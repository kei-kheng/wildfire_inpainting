# Description
This code implements a Generative Adversarial Network (GAN) to perform inpainting on [overhead infrared (IR) images of a series of prescribed fires on an area of 10 x 10 m](https://www.fs.usda.gov/rds/archive/catalog/RDS-2022-0076). The GAN has two sub-models, the context encoder (generator) and patch discriminator (discriminator). The images extracted from each video instance were limited to 100 to make the dataset more compact and diverse. This also avoids the model from biasing and overfitting. (Beyond test6) The 100 images were then split into 90 images as the training dataset and 10 images used to validate the model.

**Evaluation metrics:** MSE, PSNR, SSIM. Calculated over the **entire image**. Older implementation calculates these over the inpainted regions only, modified to be consistent with 7_environment_RGB.

- `main.py` trains the models on the training dataset and saves the models. During each epoch, the results (ground truth, masked images and inpainted images) are saved to 'results'. The training conditions are saved in a TXT file in 'models/{args.output_dir}', epoch number, losses and evaluation metrics are logged in a CSV file, plotted and saved in 'results'.
- `inference.py` loads a trained generator and performs inferencing, on dataset it has never seen during its training phase. It then writes to and plots from a CSV file (PSNR and SSIM against 'folders').
- `models.py` contains the definition of the sub-models.
- `image_utils.py` contains functions needed to preprocess the training images, write to & plot from CSVs and calculate losses/evaluation metrics.

# Usage - main.py
## Parameters
```
--data_dir          Base directory containing folders of images used to train the GAN
--folders           List of subfolder names to load training dataset from
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
--data_dir          Base directory containing folders of images used to validate the GAN
--folders           List of subfolder names to load inference dataset from
--model_path        Directory to load trained generator from
--img_scaled_dim    The dimension of the longest side of the image after scaling
--coverage          Fraction of image to be masked
--batch_size        The number of images in each mini-batch
--output_dir        Name of folder to store inference results
--num_show          Number of images to be displayed
```

**Example**
```
python inference.py --data_dir dataset/inference --folders 3 5 6 7 --model_path models/test2/generator.pth --img_scaled_dim 320 --coverage 0.15 --batch_size 5 --output_dir test2 --num_show 5
```

# Results
- test1 and test2 contain the training conditions for the currently best-known parameters, differing only in `img_scaled_dim`. It was found that a batch size of 16 yielded the shortest training time and best inpainting performance. Tested: 16, 32, 64 for a dataset size of 500.
- test3 and test4 trained models on small and big datasets respectively, where each training image is randomly rotated up to 360 degrees. This caused the model to learn useless representation of the black, unfilled regions due to rotation.
- test5 and test6 trained models on small (522) and big (903) datasets on rotated images (0, 90, 180, 270) with the appplication of more complex masks.
- **Split data -> 90% training, 10% validation (inferencing)**
- test7 validates the functionality to **calculate and plot the performance metrics - SSIM & PSNR**.
- test8-test11 trained models on the full dataset with epochs 50, 100, 150 and 200.