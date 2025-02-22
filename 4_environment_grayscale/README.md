# Description
This code loads the image of a grayscale ellipse as the environment for an agent to explore. Agent performs random walk in the environment, maintaining cumulative memory of the observation it has made. The observation is input into a trained model. Visualisation is done using Pygame. The FPS of the simulation can be modified by changing the argument of `clock.tick()`.

# Usage - main.py
## Parameters
```
--env_img           Path to image to be loaded as environment, default: images/default.png
--img_size          Size of image in pixels, default: 128x128 (width X height)
--model_path        Path to load trained odel from, default: models/inpainting_autoencoder_grayscale.pth
--agent_patch_size  Size of patch observable by agent, default: 9
--steps             Simulation time in ticks, default: 1000
```

**Example**
```
python main.py --env_img environment/default.png --img_size 128 128 --model_path models/denoising_autoencoder_grayscale.pth --agent_patch_size 9 --steps 10000
```