# Description
This directory contains 3 versions of code that generate synthetic images of colour-coded (function of fire intensity) ellipses.
- **main_with_axes_backup.py** - Oldest workable version of the code, generates a single image with labels, axes and colourbar.
- **main_single_image.py** - Generates a single image without labels, axes and colourbar.
- **main_batch.py** - Generates a batch of images with randomized maximum intensity, wind, wind speed and wind direction.

# Usage - main_single_image.py
## Parameters (optional)
```
--output          Directory to store generated image
--size            Length of generated image (square) in pixels, default: 720x720
--lim             Domain of x and y-axes, default: [-10, 10]
--ignition_point  Cartesian coordinates of fire's starting point, default: (0, 0)
--max_intensity   Maximum intensity of fire: 4, 3, 2, 1. Higher number represents greater fire intensity, default: 3
--wind            Specifies whether wind is present, default: False
--wind_speed      Speed of wind in km/h, default: 5 km/h
--wind_dir        Direction of wind in degrees, 0 degree lies on the positive x-axis, increasing counterclockwise, default: 0.0
```

**Example**
```
python main_single_image.py --output dataset/image.png --size 720 --lim -10 10 --ignition_point 0.0 0.0 --max_intensity 3 --wind --wind_speed 5.0 --wind_dir 136.0
```

# Usage - main_batch.py
## Parameters
```
--output          Directory to store generated images, default: dataset
--no_of_images    Number of images to be generated, default: 10
--size            Length of generated image (square) in pixels, default: 720x720
--lim             Domain of x and y-axes, default: [-10, 10]. --lim expects one argument
```

**Example**
```
python main_batch.py --output dataset --no_of_images 100 --size 720 --lim 10
```