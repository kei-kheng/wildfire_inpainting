# Import libraries
import argparse
import os
import random
import pygame
import numpy as np
import torch

from models import ContextEncoder
from agent import Agent
from utils import img_to_nparray, img_to_tensor, nparray_to_surface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_img_path", type=str, default="env_imgs/default.png")
    parser.add_argument("--img_scaled_dim", type=int, default=320)
    parser.add_argument("--model_path", type=str, default="models/test6/generator.pth")
    parser.add_argument("--no_of_agents", type=int, default=2)
    parser.add_argument("--agent_patch_size", type=int, default=9)
    parser.add_argument("--agent_comm_range", type=int, default=9)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="test")
    args = parser.parse_args()

    os.makedirs(f"results/{args.output_dir}/images", exist_ok=True)

    scale = 2
    col = 4

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = ContextEncoder(in_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded inference model from '{args.model_path}'")

    # Convert environment image to NumPy array with shape (H, W, 3)
    env_array = img_to_nparray(args.env_img_path, args.img_scaled_dim)
    env_surface = nparray_to_surface(env_array, scale)
    # Convert to tensor and normalize
    env_tensor = img_to_tensor(args.env_img_path, args.img_scaled_dim)
    env_h, env_w, env_c = env_array.shape
    # print(f"Environment shape (H, W, C): {env_array.shape}")

    # Initialize maps shared by the agents
    observed_map = np.zeros((env_h, env_w, env_c), dtype=np.float32)  # RGB
    # Channel number omitted, binary map
    explored_map = np.zeros((env_h, env_w), dtype=np.float32)  # Grayscale

    # Instantiate agents at runtime: agent_1, agent_2...
    # Be careful when passing 'position' -> Invert them! 
    agents = {}
    for i in range(1, args.no_of_agents + 1):
        agents[f"agent_{i}"] = Agent(
            position=(  # Pass coordinates as (y, x) -> Tested
                random.randint(0, env_h),
                random.randint(0, env_w),
            ),
            env_size=(env_h, env_w),
            patch_size=args.agent_patch_size,
            comm_range=args.agent_comm_range,
            observed=observed_map,
            explored=explored_map,
        )

    pygame.init()
    # Screen setup and clock initialisation
    window_w = env_w * scale * col
    window_h = env_h * scale
    window_font = pygame.font.SysFont("Comic Sans MS", 24)
    screen = pygame.display.set_mode((window_w, window_h))
    clock = pygame.time.Clock()

    for step in range(args.steps):
        pygame.display.set_caption("Multi-agent Exploration, t = {}".format(step))
        clock.tick(60)  # Limit FPS to 60

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # White background for screen
        screen.fill((255, 255, 255))

        # 1st column - Ground Truth
        text_surface = window_font.render("Ground Truth", False, (0, 0, 0))
        screen.blit(text_surface, (0, 0))
        screen.blit(env_surface, (0, 40))

        # 2nd column - Observation
        text_surface = window_font.render("Agents' Observation", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (1.0 / 4.0), 0))

        # Update agents' observation and perform random walk
        for i in range(1, args.no_of_agents + 1):
            agents[f"agent_{i}"].measure(env_array)

        # Observation is shared hence it makes no difference from which agent it is from
        obs_array = agents[f"agent_{args.no_of_agents}"].get_observation()
        obs_surface = nparray_to_surface(obs_array, scale)
        screen.blit(obs_surface, (window_w * (1.0 / 4.0), 40))

        # Draw agent
        for i in range(1, args.no_of_agents + 1):
            agents[f"agent_{i}"].draw(screen, scale, window_w * (1.0 / 4.0))

        # 3rd column - Explored Region
        text_surface = window_font.render("Explored Region", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (2.0 / 4.0), 0))

        exp_array = agents[f"agent_{args.no_of_agents}"].get_explored()
        # Multiplication by 255 to convert [0, 1] to [0, 255] -> Pygame's expected input format
        exp_surface = nparray_to_surface(exp_array * 255.0, scale, grayscale=True)
        screen.blit(exp_surface, (window_w * (2.0 / 4.0), 40))

        # 4th column - Prediction
        text_surface = window_font.render("Prediction", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (3.0 / 4.0), 0))

        '''
        # -> Passing observed_tensor as the model's input would give weird-coloured outputs -> Took 3 days to figure this out!
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        observed_tensor = transform(obs_array)
        observed_tensor = observed_tensor.unsqueeze(0).to(device) # Convert to tensor, normalize to [-1, 1]
        '''
        
        mask_3ch = np.repeat(exp_array[np.newaxis, :, :], 3, axis=0)  # Derive a 3-channeled mask from 'exp_array'
        mask_tensor = torch.from_numpy(mask_3ch)  # Convert to tensor, shape -> (C, H, W)
        masked_env_tensor = (env_tensor * mask_tensor).to(device)  # Masked environment

        with torch.no_grad():
            # In [-1, 1] range, shape -> (B, C, H, W)
            predicted_tensor = model(masked_env_tensor.unsqueeze(0))

        predicted_tensor = predicted_tensor.cpu().squeeze()  # Shape -> (C, H, W)

        comp_tensor = env_tensor * mask_tensor + predicted_tensor * (1 - mask_tensor)
        comp_array = comp_tensor.numpy()

        # print("comp_tensor shape:", comp_tensor.shape)
        # print("comp_array shape:", comp_array.shape)
        # print("env_tensor shape:", env_tensor.shape)
        # print("mask_tensor shape:", mask_tensor.shape)
        # print("predicted_tensor shape:", predicted_tensor.shape)

        # print("comp_tensor:", comp_tensor)
        # print("comp_array:", comp_array)
        # print("env_tensor:", env_tensor)
        # print("mask_tensor:", mask_tensor)
        # print("predicted_tensor:", predicted_tensor)
        
        # (C, H, W) to (H, W, C)
        comp_array = np.transpose(comp_array, (1, 2, 0))
        # Convert [-1, 1] to [0, 1] range
        comp_array = ((comp_array + 1.0) / 2.0)
        comp_surface = nparray_to_surface(comp_array * 255.0, scale)
        screen.blit(comp_surface, (window_w * (3.0 / 4.0), 40))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()