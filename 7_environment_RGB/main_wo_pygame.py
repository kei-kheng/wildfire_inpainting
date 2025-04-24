import os
import yaml
import argparse
import csv
import random
import numpy as np
import torch
import itertools

from models import ContextEncoder
from agent import Agent
from utils import (
    random_environment,
    convert_img_to_, 
    add_gaussian_noise_tensor,
    add_salt_and_pepper_noise_tensor,
    cal_MSE,
    cal_PSNR,
    cal_SSIM,
    plot_from_csv
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str)
    parser.add_argument("--env1_path", type=str, default=None)
    parser.add_argument("--env1_rotate", type=int, default=None)
    parser.add_argument("--env2_path", type=str, default=None)
    parser.add_argument("--env2_rotate", type=int, default=None)
    parser.add_argument("--img_scaled_dim", type=int, default=320)
    parser.add_argument("--model_path", type=str, default="models/150_epochs/run1/generator.pth")
    parser.add_argument("--no_of_agents", type=int, default=20)
    # -------------------- Depends on hardware --------------------
    parser.add_argument("--agent_patch_size", type=int, default=25)
    parser.add_argument("--agent_comm_range", type=int, default=30)
    parser.add_argument("--max_payload_size", type=int, default=90)
    # -------------------------------------------------------------
    parser.add_argument("--noise", type=str, choices=["none", "gaussian", "salt_and_pepper"], default="none")
    parser.add_argument("--agent_confidence_reception", type=float, default=0.6)
    parser.add_argument("--agent_confidence_decay", type=float, default=0.004)
    parser.add_argument("--agent_confidence_threshold", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="test")
    args = parser.parse_args()

    if args.yaml_path:
        with open(args.yaml_path, "r") as f:
            config_args = yaml.safe_load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    env_img_1_path, env_img_2_path = (
        (args.env1_path, args.env2_path) if args.env1_path and args.env2_path 
        else random_environment("env_imgs", sample=2)
    )

    random_angle_1 = args.env1_rotate if args.env1_rotate is not None else random.choice((0, 90, 180, 270))
    random_angle_2 = args.env2_rotate if args.env2_rotate is not None else random.choice((0, 90, 180, 270))

    os.makedirs(f"results/{args.output_dir}", exist_ok=True)
    
    sim_log_path = f"results/{args.output_dir}/simulation_conditions.txt"
    with open(sim_log_path, "w") as f:
        f.write(f"Initial environment: {env_img_1_path} with {random_angle_1} degree rotation\n")
        f.write(f"Switched environment: {env_img_2_path} with {random_angle_2} degree rotation\n")
        for arg_key, arg_value in vars(args).items():
            f.write(f"--{arg_key} {arg_value}\n")
    print(f"Wrote simulation conditions to: {sim_log_path}")

    csv_path = f"results/{args.output_dir}/evaluation_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "MSE", "PSNR", "SSIM", "Percentage Explored"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = ContextEncoder(in_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded inference model from '{args.model_path}'")

    env_array = convert_img_to_(env_img_1_path, args.img_scaled_dim, output="nparray", rotation=random_angle_1)
    env_tensor = convert_img_to_(env_img_1_path, args.img_scaled_dim, output="tensor", rotation=random_angle_1)
    env_h, env_w, env_c = env_array.shape

    agents = {}
    agent_keys = [f"agent_{i}" for i in range(1, args.no_of_agents + 1)]

    policy_dict = {key: "random" for key in agent_keys}

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
            sample_points=4,
            compress_mode=True
        )

    for step in range(args.steps):
        # 1st column - Ground Truth
        # Change environment at half the simulation time
        if (step == args.steps // 2):
            env_array = convert_img_to_(env_img_2_path, args.img_scaled_dim, output="nparray", rotation=random_angle_2)
            env_tensor = convert_img_to_(env_img_2_path, args.img_scaled_dim, output="tensor", rotation=random_angle_2)
            print("Environment changed!")

        # 2nd column - Observation
        # Update agents' observation, perform random walk, populate payload and update confidence
        for i in range(1, args.no_of_agents + 1):
            agents[f"agent_{i}"].measure(env_array)
            agents[f"agent_{i}"].populate_payload(args.max_payload_size)
            agents[f"agent_{i}"].update_confidence()
        
        obs_array = agents["agent_1"].get_observation()
        obs_tensor = torch.from_numpy(obs_array).permute(2, 0, 1).float()
        obs_tensor = (obs_tensor / 255.0 - 0.5) * 2.0

        explored_map = agents["agent_1"].get_explored()
        percentage_explored = np.sum(explored_map) / explored_map.size * 100

        agent_names = list(agents.keys())
        for i, j in itertools.combinations(agent_names, 2):
            agent_i = agents[i]
            agent_j = agents[j]

            pos_i = np.array(agent_i.get_position())
            pos_j = np.array(agent_j.get_position())

            distance = np.linalg.norm(pos_i - pos_j)

            if distance <= args.agent_comm_range:
                agent_i.communicate_with(agent_j, confidence_reception=args.agent_confidence_reception)
                agent_j.communicate_with(agent_i, confidence_reception=args.agent_confidence_reception)

        exp_array = agents["agent_1"].get_explored()
        mask_3ch = np.repeat(exp_array[np.newaxis, :, :], 3, axis=0)
        mask_tensor = torch.from_numpy(mask_3ch)

        if args.noise == "gaussian":
            obs_tensor = add_gaussian_noise_tensor(obs_tensor)
        elif args.noise == "salt_and_pepper":
            obs_tensor = add_salt_and_pepper_noise_tensor(obs_tensor)
        
        masked_env_tensor = (obs_tensor * mask_tensor).to(device)

        with torch.no_grad():
            predicted_tensor = model(masked_env_tensor.unsqueeze(0))

        predicted_tensor = predicted_tensor.cpu().squeeze()
        comp_tensor = obs_tensor * mask_tensor + predicted_tensor * (1 - mask_tensor)
        comp_array = comp_tensor.numpy()
        comp_array = np.transpose(comp_array, (1, 2, 0))
        comp_array = ((comp_array + 1.0) / 2.0)    

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

    print("Simulation Done!")

if __name__ == "__main__":
    main()