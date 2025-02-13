import numpy as np
import random
import pygame

class Agent:
    def __init__(self, start_pos, map_size, patch_size):
        self.position = start_pos
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.patch_size = patch_size
        # Size of agent in pixels
        self.agent_size = 10
        # Maps storing information of what the agent has seen and regions it have explored
        self.observed = np.zeros((self.map_width, self.map_height))
        self.explored = np.zeros((self.map_width, self.map_height))
        self.dy, self.dx = 0, 1 # Default: Move to the right

    # Change moving direction with 30% probability
    def random_walk(self):
        if random.random() < 0.7:
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
            self.dy, self.dx = random.choice(dirs)

        pos_x, pos_y = self.position
        pos_x += self.dx
        pos_y += self.dy

        # Clamp values to ensure valid indices
        # self.agent_size//2 - integer division, don't want agent's icon to be drawn outside boundaries
        pos_x = max(self.agent_size//2, min(self.map_width-1-self.agent_size//2, pos_x))
        pos_y = max(self.agent_size//2, min(self.map_height-1-self.agent_size//2, pos_y))
        self.position = (pos_x, pos_y)

    # Make an observation, update 'observed' and 'explored' given the NumPy array of the environment
    # Then, perform random walk
    def measure(self, env_array):
        pos_x, pos_y = self.position
        observe_radius = self.patch_size//2

        # Clamp to ensure valid indices
        x0 = max(0, pos_x - observe_radius)
        y0 = max(0, pos_y - observe_radius)
        x1 = min(self.map_width - 1, pos_x + observe_radius)
        y1 = min(self.map_height - 1, pos_y + observe_radius)

        # Extract patch from env_array (grayscale environment, (H, W)) -> Need to change to (H, W, 3) for RGB image
        patch = env_array[x0:x1+1, y0:y1+1] # '+1' due to how array slicing works in Python

        # Update agent's maps
        self.observed[x0:x1+1, y0:y1+1] = patch
        self.explored[x0:x1+1, y0:y1+1] = 1
        self.random_walk()

    def get_position(self):
        return self.position

    def get_observation(self):
        return self.observed

    def get_explored(self):
        # 1: Explored, 0: Unknown
        return self.explored
    
    # Draw agent
    def draw(self, screen, scale, x_coord):
        # (R, G, B), (x, y, width, height)
        pygame.draw.rect(screen, (255, 0, 0), (self.position[1] * scale + x_coord, self.position[0] * scale + 30, self.agent_size, self.agent_size))
        pygame.draw.circle(screen, (255, 255, 255), (self.position[1] * scale + x_coord + 7, self.position[0] * scale + 32), 2)
        pygame.draw.circle(screen, (255, 255, 255), (self.position[1] * scale + x_coord + 2, self.position[0] * scale + 32), 2)
        pygame.draw.circle(screen, (0, 0, 0), (self.position[1] * scale + x_coord + 7, self.position[0] * scale + 32), 1)
        pygame.draw.circle(screen, (0, 0, 0), (self.position[1] * scale + x_coord + 2, self.position[0] * scale + 32), 1)