import random
import pygame

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
    def __init__(self, position, env_size, patch_size, comm_range, observed, explored):
        self.position = position
        self.env_w = env_size[0]
        self.env_h = env_size[1]
        self.patch_size = patch_size
        self.comm_range = comm_range
        self.agent_size = 10

        # Maps storing information of what the agents have seen and regions they have explored
        self.observed = observed  # 3 channels
        self.explored = explored  # 1: Explored, 0: Unknown

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
        self.explored[x0 : x1 + 1, y0 : y1 + 1] = 1
        self.random_walk()

    def get_position(self):
        return self.position

    def get_observation(self):
        return self.observed

    def get_explored(self):
        return self.explored

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
