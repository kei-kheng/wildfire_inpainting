import argparse
import pygame
import numpy as np
import random
from PIL import Image

import torch

from autoencoder import ConvAutoencoder
from agent import Agent

def array_to_surface(nparray, scale, multiply_255=False):
    # pygame expects 3 channels (RGB), hence, stack 3 times along last axis -> (H, W, 3)
    img_3ch = np.stack((nparray, nparray, nparray), axis=-1)
    # Scale the image using Kronecker product
    img_scaled = np.kron(img_3ch, np.ones((scale,scale,1)))
    # Convert array to 8-bit integers (0-255)
    if multiply_255==True:
        img_scaled = (img_scaled*255).astype(np.uint8)
    else:
        img_scaled = img_scaled.astype(np.uint8)
    # Reorder from (H, W, 3) to (W, H, 3), then make surface
    img_surface = pygame.surfarray.make_surface(np.transpose(img_scaled,(1,0,2)))
    return img_surface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_img", type=str, default="environment/default.png")
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--model_path", type=str, default="models/inpainting_autoencoder_grayscale.pth")
    parser.add_argument("--agent_patch_size", type=int, default=9)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: '{device}'")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded model from '{args.model_path}'")

    # Initialize 'observed' and 'explored' maps of multi-agent system
    observed_map = np.zeros((args.img_size[0], args.img_size[1]), dtype=np.float32)
    explored_map = np.zeros((args.img_size[0], args.img_size[1]), dtype=np.float32)

    # Instantiate multiple agents at random positions
    agent1 = Agent(start_pos=(random.randint(0, args.img_size[0]), random.randint(0, args.img_size[1])),
                   map_size=(args.img_size[0],args.img_size[1]),
                   patch_size=args.agent_patch_size,
                   singleAgent=False,
                   observed=observed_map,
                   explored=explored_map)
    
    agent2 = Agent(start_pos=(random.randint(0, args.img_size[0]), random.randint(0, args.img_size[1])),
                   map_size=(args.img_size[0],args.img_size[1]),
                   patch_size=args.agent_patch_size,
                   singleAgent=False,
                   observed=observed_map,
                   explored=explored_map)
    
    agent3 = Agent(start_pos=(random.randint(0, args.img_size[0]), random.randint(0, args.img_size[1])),
                   map_size=(args.img_size[0],args.img_size[1]),
                   patch_size=args.agent_patch_size,
                   singleAgent=False,
                   observed=observed_map,
                   explored=explored_map)

    # Initialize imported modules
    pygame.init()

    # Scale image by some factor
    scale = 5
    # Number of columns to display
    col = 4
    w_width = args.img_size[0] * scale * col
    w_height = args.img_size[1] * scale
    w_font = pygame.font.SysFont("Comic Sans MS", 24)

    # Set window size and name
    screen = pygame.display.set_mode((w_width, w_height))
    # Want to use clock to limit FPS
    clock = pygame.time.Clock()

    # Ground Truth - can be outside for loop because Ground Truth does not change, but needs to be blitted in the loop
    # Returns (H, W, 1), not (W, H)!
    env_img = Image.open(args.env_img).convert("L")
    # Reference: https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/
    env_array = np.array(env_img)
    env_surface = array_to_surface(env_array, 5)

    for step in range(args.steps):
        pygame.display.set_caption("Single-Agent Exploration: Grayscale t = {}".format(step))
        clock.tick(40) # Limit FPS

        for event in pygame.event.get():
             if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        screen.fill((255, 255, 255))

        # 1st column - Ground Truth
        text_surface = w_font.render("Ground Truth", False, (0, 0, 0))
        screen.blit(text_surface, (0, 0))
        screen.blit(env_surface, (0, 40))

        # 2nd column - Observation
        text_surface = w_font.render("Observation", False, (0, 0, 0))
        screen.blit(text_surface, (w_width * (1.0/4.0), 0))

        # Update agents' observation and perform random walk
        agent1.measure(env_array)
        agent2.measure(env_array)
        agent3.measure(env_array)

        observed_array = agent3.get_observation()
        observed_surface = array_to_surface(observed_array, 5)
        screen.blit(observed_surface, (w_width * (1.0/4.0), 40))

        agent1.draw(screen, 5, w_width * (1.0/4.0))
        agent2.draw(screen, 5, w_width * (1.0/4.0))
        agent3.draw(screen, 5, w_width * (1.0/4.0))

        # 3rd column - Explored Region
        text_surface = w_font.render("Explored Region", False, (0, 0, 0))
        screen.blit(text_surface, (w_width * (2.0/4.0), 0))

        explored_array = agent3.get_explored()
        # Multiplication by 255 here is necessary. Else, the surface appears entirely black throughout the simulation.
        explored_surface = array_to_surface(explored_array, 5, multiply_255=True)
        screen.blit(explored_surface, (w_width * (2.0/4.0), 40))

        # 4th column - Prediction
        text_surface = w_font.render("Prediction", False, (0, 0, 0))
        screen.blit(text_surface, (w_width * (3.0/4.0), 0))

        # Convert observation_array to tensor (float32)
        # Reference: https://stackoverflow.com/questions/74848349/pytorch-runtime-error-input-type-double-and-bias-type-float-should-be-the-s
        observed_array_copy = observed_array.astype(np.float32)
        observed_tensor = torch.from_numpy(observed_array_copy)
        # Model expects (1, 1, H, W)
        observed_tensor = observed_tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_tensor = model(observed_tensor)

        # Move to CPU, convert to (H, W) NumPy array
        predicted_array = predicted_tensor.cpu().numpy().squeeze()
        predicted_surface = array_to_surface(predicted_array, 5, multiply_255=True)
        screen.blit(predicted_surface, (w_width * (3.0/4.0), 40))

        pygame.display.flip()
    
    pygame.quit()

if __name__=="__main__":
    main()