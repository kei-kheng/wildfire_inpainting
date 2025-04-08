# Import libraries
import os
import argparse
import csv
import random
import pygame
import numpy as np
import torch
import torchvision.transforms as T
import itertools # For distance measurement between agents

from models import ContextEncoder
from agent import Agent
from utils import (
    random_environment,
    convert_img_to_, 
    nparray_to_surface,
    cal_MSE,
    cal_PSNR,
    cal_SSIM,
    plot_from_csv
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_scaled_dim", type=int, default=320)
    parser.add_argument("--model_path", type=str, default="models/test8/generator.pth")
    parser.add_argument("--no_of_agents", type=int, default=20)
    # -------------------- Depends on hardware --------------------
    parser.add_argument("--agent_patch_size", type=int, default=25)
    parser.add_argument("--agent_comm_range", type=int, default=30)
    parser.add_argument("--max_payload_size", type=int, default=270)
    # -------------------------------------------------------------
    parser.add_argument("--agent_confidence_reception", type=float, default=0.6)
    parser.add_argument("--agent_confidence_decay", type=float, default=0.01)
    parser.add_argument("--agent_confidence_threshold", type=float, default=0.15)
    parser.add_argument("--agent_policy", type=str, default="random")
    parser.add_argument("--agent_sample_points", type=int, default=4)
    parser.add_argument("--log_comm", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="test")
    args = parser.parse_args()

    env_img_1_path, env_img_2_path = random_environment("env_imgs", sample=2)

    os.makedirs(f"results/{args.output_dir}", exist_ok=True)
    # Option to log communication
    if args.log_comm:
        comm_log_path = f"results/{args.output_dir}/communication_log.csv"
        with open(comm_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Agent A", "A's Position", "Agent B", "B's Position"])
    
    # Log simulation conditions
    sim_log_path = f"results/{args.output_dir}/simulation_conditions.txt"
    with open(sim_log_path, "w") as f:
        f.write(f"Initial environment: {env_img_1_path}\n")
        f.write(f"Switched environment: {env_img_2_path}\n")
        for arg_key, arg_value in vars(args).items():
            f.write(f"--{arg_key} {arg_value}\n")
    print(f"Wrote simulation conditions to: {sim_log_path}")

    # Write to CSV file
    csv_path = f"results/{args.output_dir}/evaluation_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "MSE", "PSNR", "SSIM", "Percentage Explored"])

    # Variables for Pygame window
    scale = 2
    col = 4
    x_offset = 5.0
    y_offset = 40.0
    legend_size = 15

    # For communication between agents
    bytes_per_pixel = 3  # RGB, uint8
    max_pixels = args.max_payload_size // bytes_per_pixel
    print(f"Agents could transmit up to {max_pixels} pixels")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = ContextEncoder(in_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded inference model from '{args.model_path}'")

    # Convert environment image to NumPy array with shape (H, W, 3), random rotation applied
    random_angle_1 = random.choice((0, 90, 180, 270))
    random_angle_2 = random.choice((0, 90, 180, 270))
    env_array = convert_img_to_(env_img_1_path, args.img_scaled_dim, output="nparray", rotation=random_angle_1)
    env_surface = nparray_to_surface(env_array, scale)
    # Convert to tensor and normalize
    env_tensor = convert_img_to_(env_img_1_path, args.img_scaled_dim, output="tensor", rotation=random_angle_1)
    env_h, env_w, env_c = env_array.shape
    # print(f"Environment shape (H, W, C): {env_array.shape}")

    # Instantiate agents at runtime: agent_1, agent_2...
    # Each agent accesses and updates their own maps only
    agents = {}
    agent_keys = [f"agent_{i}" for i in range(1, args.no_of_agents + 1)]

    if args.agent_policy == "mixed":
        random.shuffle(agent_keys)
        half = len(agent_keys) // 2
        # agent: policy dict
        policy_dict = {
            key: "random" if i < half else "confidence"
            for i, key in enumerate(agent_keys)
        }
    else:
        policy_dict = {key: args.agent_policy for key in agent_keys}

    for agent_key in agent_keys:
        observed_map = np.zeros((env_h, env_w, env_c), dtype=np.float32)
        explored_map = np.zeros((env_h, env_w), dtype=np.float32)
        confidence_matrix = np.zeros((env_h, env_w), dtype=np.float32)

        agents[agent_key] = Agent(
            position=(
                random.randint(0, env_w),
                random.randint(0, env_h),
            ),
            env_size=(env_w, env_h),
            patch_size=args.agent_patch_size,
            comm_range=args.agent_comm_range,
            observed=observed_map,
            explored=explored_map,
            confidence=confidence_matrix,
            confidence_decay=args.agent_confidence_decay,
            confidence_threshold=args.agent_confidence_threshold,
            policy=policy_dict[agent_key],
            sample_points=args.agent_sample_points
        )
    
    # Choose 4 agents besides agent_1 randomly whose maps are to be displayed
    if args.no_of_agents > 1:
        agent_keys.remove("agent_1")
        displayed_agents_list = random.sample(agent_keys, 4)
        displayed_agents_dict = {}
        agent_colour = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]
        for agent, colour in zip(displayed_agents_list, agent_colour):
            displayed_agents_dict[agent] = colour

    pygame.init()
    # Screen setup and clock initialisation
    window_w = env_w * scale * col
    if args.no_of_agents > 1:
        window_h = (2 * env_h + y_offset) * scale
    else:
        window_h = (env_h + y_offset) * scale
    window_font = pygame.font.SysFont("Comic Sans MS", 24)
    screen = pygame.display.set_mode((window_w, window_h))
    clock = pygame.time.Clock()

    for step in range(args.steps):
        pygame.display.set_caption("Multi-agent Exploration, t = {}".format(step))
        clock.tick(120)  # Limit FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # White background for screen
        screen.fill((255, 255, 255))

        # 1st column - Ground Truth
        # Change environment at half the simulation time
        if (step == args.steps // 2):
            env_array = convert_img_to_(env_img_2_path, args.img_scaled_dim, output="nparray", rotation=random_angle_2)
            env_surface = nparray_to_surface(env_array, scale)
            env_tensor = convert_img_to_(env_img_2_path, args.img_scaled_dim, output="tensor", rotation=random_angle_2)
            print("Environment changed!")

        text_surface = window_font.render("Ground Truth", False, (0, 0, 0))
        screen.blit(text_surface, (x_offset, 0))
        screen.blit(env_surface, (0, y_offset))

        # 2nd column - Observation
        # Update agents' observation, perform random walk, populate payload and update confidence
        for i in range(1, args.no_of_agents + 1):
            agents[f"agent_{i}"].measure(env_array)
            agents[f"agent_{i}"].populate_payload(max_pixels)
            agents[f"agent_{i}"].update_confidence()
        
        obs_array = agents["agent_1"].get_observation()
        obs_surface = nparray_to_surface(obs_array, scale)
        # (H, W, C) to (C, H, W), use permute() for tensors and transpose() for NumPy arrays
        obs_tensor = torch.from_numpy(obs_array).permute(2, 0, 1).float()
        # Normalize to [-1.0, 1.0] from [0, 255] -> Did a thought experiment with 0 and 255
        obs_tensor = (obs_tensor / 255.0 - 0.5) * 2.0
        screen.blit(obs_surface, (window_w * (1.0 / 4.0), y_offset))

        explored_map = agents["agent_1"].get_explored()
        percentage_explored = np.sum(explored_map) / explored_map.size * 100

        text_surface = window_font.render(f"Agent 1's Observation ({percentage_explored:.2f}%)", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (1.0 / 4.0) + x_offset, 0))

        # Measure Pythagorean distance and update maps if close enough
        agent_names = list(agents.keys())
        '''
        Reference: https://docs.python.org/3/library/itertools.html#itertools.combinations
        Form and iterate through unique, two-agent-subsets from the list
        '''
        for i, j in itertools.combinations(agent_names, 2):
            agent_i = agents[i]
            agent_j = agents[j]

            pos_i = np.array(agent_i.get_position())
            pos_j = np.array(agent_j.get_position())

            # Euclidean/Pythagorean distance
            # Reference: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            distance = np.linalg.norm(pos_i - pos_j)

            # If close enough, exchange information
            if distance <= args.agent_comm_range:
                agent_i.communicate_with(agent_j, confidence_reception=args.agent_confidence_reception)
                agent_j.communicate_with(agent_i, confidence_reception=args.agent_confidence_reception)

                # Draw white line between communicating agents
                x1, y1 = agent_i.get_position()
                x2, y2 = agent_j.get_position()
                pygame.draw.line(
                    screen,
                    (255, 255, 255),
                    (x1 * scale + window_w * (1.0 / 4.0), y1 * scale + 30),
                    (x2 * scale + window_w * (1.0 / 4.0), y2 * scale + 30),
                    1
                )

                if args.log_comm:
                    with open(comm_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # Step / Agent A / A's Position / Agent B / B's Position
                        writer.writerow([
                            step, i, agent_i.get_position(), j, agent_j.get_position()
                        ])

        for i in range(1, args.no_of_agents + 1):
            agent_key = f"agent_{i}"
            if agent_key == "agent_1":
                colour = (255, 165, 0)
            elif agent_key in displayed_agents_dict:
                colour = displayed_agents_dict[agent_key]
            else:
                colour = (255, 0, 0)
            agents[agent_key].draw(screen, scale, window_w * (1.0 / 4.0), colour)

        # 3rd column - Confidence Matrix
        text_surface = window_font.render("Agent 1's Confidence Matrix", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (2.0 / 4.0) + x_offset, 0))

        # Old implementation - Showing explored areas
        '''
        # Multiplication by 255 to convert [0, 1] to [0, 255] -> Pygame's expected input format
        exp_surface = nparray_to_surface(exp_array * 255.0, scale, grayscale=True)
        screen.blit(exp_surface, (window_w * (2.0 / 4.0), y_offset))
        '''
        confidence_array = agents["agent_1"].get_confidence()
        confidence_surface = nparray_to_surface(confidence_array * 255.0, scale, grayscale=True)
        screen.blit(confidence_surface, (window_w * (2.0 / 4.0), y_offset))

        # 4th column - Prediction
        # Legend
        pygame.draw.rect(screen, (255, 165, 0), pygame.Rect(
                window_w * (3.0 / 4.0) + x_offset,
                10.0,
                legend_size, legend_size
            )
        )
        agent_policy = agents["agent_1"].get_policy()
        text_surface = window_font.render("Agent 1's Prediction", False, (0, 0, 0))
        # text_surface = window_font.render(f"Agent 1's Prediction, policy: {agent_policy}", False, (0, 0, 0))
        screen.blit(text_surface, (window_w * (3.0 / 4.0) + 2 * x_offset + legend_size, 0))

        '''
        # Passing observed_tensor as the model's input without multiplication by mask would give weird-coloured outputs
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        observed_tensor = transform(obs_array)
        observed_tensor = observed_tensor.unsqueeze(0).to(device) # Convert to tensor, normalize to [-1, 1]
        '''
        
        exp_array = agents["agent_1"].get_explored()
        mask_3ch = np.repeat(exp_array[np.newaxis, :, :], 3, axis=0)  # Derive a 3-channeled mask from 'exp_array'
        mask_tensor = torch.from_numpy(mask_3ch)  # Convert to tensor, shape -> (C, H, W)
        # masked_env_tensor = (env_tensor * mask_tensor).to(device)  # For static environment
        masked_env_tensor = (obs_tensor * mask_tensor).to(device)

        with torch.no_grad():
            # In [-1, 1] range, shape -> (B, C, H, W)
            predicted_tensor = model(masked_env_tensor.unsqueeze(0))

        predicted_tensor = predicted_tensor.cpu().squeeze()  # Shape -> (C, H, W)

        # comp_tensor = env_tensor * mask_tensor + predicted_tensor * (1 - mask_tensor) # For static environment
        comp_tensor = obs_tensor * mask_tensor + predicted_tensor * (1 - mask_tensor)
        comp_array = comp_tensor.numpy()

        # print("comp_tensor shape:", comp_tensor.shape)
        # print("comp_array shape:", comp_array.shape)
        # print("env_tensor shape:", env_tensor.shape)
        # print("obs_tensor shape:", obs_tensor.shape)
        # print("mask_tensor shape:", mask_tensor.shape)
        # print("predicted_tensor shape:", predicted_tensor.shape)

        # print("comp_tensor:", comp_tensor)
        # print("comp_array:", comp_array)
        # print("env_tensor:", env_tensor)
        # print("obs_tensor shape:", obs_tensor)
        # print("mask_tensor:", mask_tensor)
        # print("predicted_tensor:", predicted_tensor)
        
        # (C, H, W) to (H, W, C)
        comp_array = np.transpose(comp_array, (1, 2, 0))
        # Convert [-1, 1] to [0, 1] range
        comp_array = ((comp_array + 1.0) / 2.0)
        comp_surface = nparray_to_surface(comp_array * 255.0, scale)
        screen.blit(comp_surface, (window_w * (3.0 / 4.0), y_offset))

        # 2nd row - Display observation of agents in displayed_agents_list
        if args.no_of_agents > 1:
            for idx, d_agent_key in enumerate(displayed_agents_list):
                pygame.draw.rect(screen, displayed_agents_dict[d_agent_key], pygame.Rect(
                    window_w * (idx / 4.0) + x_offset,
                    env_h * scale + y_offset + 10.0,
                    legend_size, legend_size
                    )
                )
                # agent_policy = agents[d_agent_key].get_policy()
                text_surface = window_font.render(f"Prediction ({d_agent_key})", False, (0, 0, 0))
                screen.blit(text_surface, (window_w * (idx / 4.0) + 2 * x_offset + legend_size, env_h * scale + y_offset))

                d_explored = agents[d_agent_key].get_explored()
                d_observed_array = agents[d_agent_key].get_observation()
                d_obs_tensor = torch.from_numpy(d_observed_array).permute(2, 0, 1).float()
                d_obs_tensor = (d_obs_tensor / 255.0 - 0.5) * 2.0
                d_mask = np.repeat(d_explored[np.newaxis, :, :], 3, axis=0)
                d_mask_tensor = torch.from_numpy(d_mask)
                # d_masked_env_tensor = (env_tensor * d_mask_tensor).to(device) # For static environment
                d_masked_env_tensor = (d_obs_tensor * d_mask_tensor).to(device)

                with torch.no_grad():
                    d_predicted_tensor = model(d_masked_env_tensor.unsqueeze(0))
                d_predicted_tensor = d_predicted_tensor.cpu().squeeze()

                d_comp_tensor = d_obs_tensor * d_mask_tensor + d_predicted_tensor * (1 - d_mask_tensor)
                d_comp_array = d_comp_tensor.numpy()
                d_comp_array = np.transpose(d_comp_array, (1, 2, 0))
                d_comp_array = ((d_comp_array + 1.0) / 2.0)
                d_comp_surface = nparray_to_surface(d_comp_array * 255.0, scale)
                screen.blit(d_comp_surface, (window_w * (idx / 4.0), env_h * scale + 2 * y_offset))    

        # Calculate PSNR and SSIM, write to CSV
        if step % 10 == 0:
            comp_copy = comp_tensor.cpu().numpy().copy()
            env_copy = env_tensor.cpu().numpy().copy()

            MSE_val = cal_MSE(comp_copy, env_copy)
            PSNR_val = cal_PSNR(comp_copy, env_copy)
            SSIM_val = cal_SSIM(comp_copy, env_copy)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                # Step / MSE / PNSR / SSIM / Percentage Explored
                writer.writerow([
                    step, 
                    f"{MSE_val:.4f}",
                    f"{PSNR_val:.4f}", 
                    f"{SSIM_val:.4f}",
                    f"{percentage_explored:.2f}"
                ])
            
        if step % 100 == 0:
            save_path = f"results/{args.output_dir}/step_{step}.png"
            pygame.image.save(screen, save_path)
            # print(f"Step = {step}, PSNR = {PSNR_val:.4f}, SSIM = {SSIM_val:.4f}")
            print(f"Saved image to: {save_path}")

        pygame.display.flip()

    pygame.quit()

    plot_from_csv(args.output_dir, csv_file="evaluation_log.csv")
    print("Plotted graphs from CSV")

if __name__ == "__main__":
    main()