import random
import pygame
import numpy as np

"""
Agent's attributes:
- Position
- Size of environment (Width, Height)
- Patch size
- Communication range
- Observation
- Explored region
"""


class Agent:
    def __init__(self, position, env_size, patch_size, comm_range, observed, explored, confidence):
        self.position = position
        self.env_w = env_size[0]
        self.env_h = env_size[1]
        self.patch_size = patch_size
        self.comm_range = comm_range
        self.agent_size = 10

        # Maps storing information of what the agents have seen and regions they have explored
        self.observed = observed  # 3 channels
        self.explored = explored  # 1: Explored, 0: Unknown

        # To model information decay/data freshness
        self.confidence = confidence

        # Default: Move to the right
        self.dy, self.dx = 0, 3

    # Change moving direction with 30% probability
    def random_walk(self):
        if random.random() < 0.7:
            dirs = [(3, 0), (-3, 0), (0, 3), (0, -3)]
            self.dy, self.dx = random.choice(dirs)

        pos_x, pos_y = self.position
        pos_x += self.dx
        pos_y += self.dy

        # Clamp to ensure valid indices
        pos_x = max(
            self.agent_size // 2, min(self.env_w - 1 - self.agent_size // 2, pos_x)
        )
        pos_y = max(
            self.agent_size // 2, min(self.env_h - 1 - self.agent_size // 2, pos_y)
        )
        self.position = (pos_x, pos_y)

    # Make an observation, update maps and perform random walk
    def measure(self, env_array):
        pos_x, pos_y = self.position
        observe_radius = self.patch_size // 2

        # Clamp to ensure valid indices
        x0 = max(0, pos_x - observe_radius)
        y0 = max(0, pos_y - observe_radius)
        x1 = min(self.env_w - 1, pos_x + observe_radius)
        y1 = min(self.env_h - 1, pos_y + observe_radius)

        # '+1' due to how array slicing works in Python
        patch = env_array[x0 : x1 + 1, y0 : y1 + 1, :]

        # Update agent's maps
        self.observed[x0 : x1 + 1, y0 : y1 + 1, :] = patch
        self.explored[x0 : x1 + 1, y0 : y1 + 1] = 1.0
        self.random_walk()
    
    @staticmethod
    # Does not depend on any instance attributes
    def average_obs(obs_1, exp_1, obs_2, exp_2):
        # Calculate weighted sum of obs uisng exp as weights
        weighted_obs = obs_1 * exp_1 + obs_2 * exp_2
        sum_exp = exp_1 + exp_2
        # Temporarily set unexplored regions 0 -> 1 to avoid division by 0 error
        # Positions where sum_exp > 0 are explored regions
        sum_exp[sum_exp == 0] = 1
        averaged_obs = weighted_obs / sum_exp
        return averaged_obs

    def communicate_with(self, other_agent):
        obs_self = self.get_observation()
        obs_other = other_agent.get_observation()
        exp_self = self.get_explored()
        exp_other = other_agent.get_explored()

        # Add channel axis at end (H, W, 1) for broadcast-multiplication between exp and obs
        # wca = with channel axis
        exp_self_wca = exp_self[..., None]
        exp_other_wca = exp_other[..., None]

        averaged_obs = Agent.average_obs(obs_self, exp_self_wca, obs_other, exp_other_wca)
        combined_exp = np.logical_or(exp_self, exp_other).astype(np.float32)

        # Update both agents, .copy() to avoid accidental shared memory
        self.set_observation(averaged_obs.copy())
        other_agent.set_observation(averaged_obs.copy())
        self.set_explored(combined_exp.copy())
        other_agent.set_explored(combined_exp.copy())

    def get_position(self):
        return self.position

    def get_observation(self):
        return self.observed

    def get_explored(self):
        return self.explored
    
    def set_observation(self, observed):
        self.observed = observed

    def set_explored(self, explored):
        self.explored = explored

    # Draw agent, x_offset is provided for the agents to be drawn in the right column of the window
    def draw(self, screen, scale, x_offset):
        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (
                self.position[1] * scale + x_offset,
                self.position[0] * scale + 30,
                self.agent_size,
                self.agent_size,
            ),
        )
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (self.position[1] * scale + x_offset + 7, self.position[0] * scale + 32),
            2,
        )
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (self.position[1] * scale + x_offset + 2, self.position[0] * scale + 32),
            2,
        )
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (self.position[1] * scale + x_offset + 7, self.position[0] * scale + 32),
            1,
        )
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (self.position[1] * scale + x_offset + 2, self.position[0] * scale + 32),
            1,
        )
