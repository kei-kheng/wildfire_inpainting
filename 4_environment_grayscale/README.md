# Description
This code loads the image of a grayscale ellipse as the environment for an agent/agents to explore. Agent performs random walk in the environment, maintaining cumulative memory of the observation it has made. When `no_of_agents > 1`, all agents contribute to update the system's observation.

The observation is input into a trained model. Visualisation is done using Pygame. The FPS of the simulation can be modified by changing the argument of `clock.tick()`.

# Usage - main.py
## Parameters
```
--env_img           Path to image to be loaded as environment, default: images/default.png
--img_size          Size of image in pixels, default: 128x128 (width X height)
--model_path        Path to load trained model from, default: models/denoising_autoencoder_grayscale.pth
--no_of_agents      Number of agents to be instantiated in the environment, default: 1
--agent_patch_size  Size of patch observable by agent, default: 9
--steps             Simulation time in ticks, default: 1000
```

**Example**
```
python main.py --env_img environment/default.png --img_size 128 128 --model_path models/partial_cae_100_epochs.pth --no_of_agents 10 --agent_patch_size 9 --steps 10000
```