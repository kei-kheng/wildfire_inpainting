import random
import pygame
import numpy as np

SPEED = 5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]
AGENT_SIZE = 10
# For confidence-based walk
LOOKAHEAD_DISTANCE = 15
SAMPLE_RADIUS = 5
OFFSET = 15

class Agent:
    def __init__(self, position, env_size, patch_size, comm_range, observed, explored, confidence, confidence_decay, confidence_threshold, policy, sample_points):
        self.position = position  # x, y
        self.env_w = env_size[0]
        self.env_h = env_size[1]
        self.patch_size = patch_size
        self.comm_range = comm_range
        self.payload = None

        # Maps storing information of what the agents have seen and regions they have explored
        self.observed = observed  # 3 channels
        self.explored = explored  # 1: Explored, 0: Unknown

        # To model information decay/data freshness
        self.confidence = confidence
        self.confidence_decay = confidence_decay
        self.confidence_threshold = confidence_threshold

        self.policy = policy

        if self.policy == "confidence":
            if sample_points == 4:
                self.sample_targets = [
                    (SAMPLE_RADIUS + OFFSET, SAMPLE_RADIUS + OFFSET),
                    (self.env_w - SAMPLE_RADIUS - OFFSET, self.env_h - SAMPLE_RADIUS - OFFSET),
                    (SAMPLE_RADIUS + OFFSET, self.env_h - SAMPLE_RADIUS - OFFSET),
                    (self.env_w - SAMPLE_RADIUS - OFFSET, SAMPLE_RADIUS + OFFSET),
                ]
            elif sample_points == 8:
                self.sample_targets = [
                    # Corners
                    (SAMPLE_RADIUS + OFFSET, SAMPLE_RADIUS + OFFSET),
                    (self.env_w - SAMPLE_RADIUS - OFFSET, self.env_h - SAMPLE_RADIUS - OFFSET),
                    (SAMPLE_RADIUS + OFFSET, self.env_h - SAMPLE_RADIUS - OFFSET),
                    (self.env_w - SAMPLE_RADIUS - OFFSET, SAMPLE_RADIUS + OFFSET),
                    # Midpoints of each edge
                    (self.env_w // 2, SAMPLE_RADIUS + OFFSET),
                    (self.env_w // 2, self.env_h - SAMPLE_RADIUS - OFFSET),
                    (SAMPLE_RADIUS + OFFSET, self.env_h // 2),
                    (self.env_w - SAMPLE_RADIUS - OFFSET, self.env_h // 2),
                ]

        # Default: Move to the right
        self.dx = SPEED
        self.dy = 0

    # Change moving direction with 30% probability
    def random_walk(self):
        if random.random() < 0.7:
            self.dx, self.dy = random.choice(DIRECTIONS)

        pos_x, pos_y = self.position
        pos_x += self.dx * SPEED
        pos_y += self.dy * SPEED

        # Clamp to ensure valid indices
        pos_x = max(AGENT_SIZE // 2, min(self.env_w - 1 - AGENT_SIZE // 2, pos_x))
        pos_y = max(AGENT_SIZE // 2, min(self.env_h - 1 - AGENT_SIZE // 2, pos_y))
        self.position = (pos_x, pos_y)

    # Confidence-based walk policy
    # Implementation 1
    '''
    # Sample a patch in each direction, patch with lowest average confidence defines best direction
    def confidence_based_walk(self):        
        pos_x, pos_y = self.position
        best_dir = random.choice(DIRECTIONS)
        lowest_conf = float("inf")

        for dx, dy in DIRECTIONS:
            # Center of patches that we sample in each direction
            nx, ny = pos_x + dx * LOOKAHEAD_DISTANCE, pos_y + dy * LOOKAHEAD_DISTANCE
            # If out of bounds, skip this direction
            if not ((0 <= nx < self.env_w) and (0 <= ny < self.env_h)):
                continue

            # Ensure valid indices when extracting confidence patch, +1 for correct array slicing
            patch = self.confidence[
                max(dy - SAMPLE_RADIUS, 0):min(dy + SAMPLE_RADIUS + 1, self.env_h),
                max(dx - SAMPLE_RADIUS, 0):min(dx + SAMPLE_RADIUS + 1, self.env_w)
            ]
            mean_conf = np.mean(patch)
            if mean_conf <= lowest_conf:
                lowest_conf = mean_conf
                best_dir = (dx, dy)

        pos_x += best_dir[0] * SPEED
        pos_y += best_dir[1] * SPEED
        pos_x = max(AGENT_SIZE // 2, min(self.env_w - 1 - AGENT_SIZE // 2, pos_x))
        pos_y = max(AGENT_SIZE // 2, min(self.env_h - 1 - AGENT_SIZE // 2, pos_y))
        self.position = (pos_x, pos_y)
    '''

    # Implementation 2
    def confidence_based_walk(self):        
        pos_x, pos_y = self.position
        best_target = None
        lowest_conf = float("inf")

        for tx, ty in self.sample_targets:
            patch = self.confidence[
                max(ty - SAMPLE_RADIUS, 0):min(ty + SAMPLE_RADIUS + 1, self.env_h),
                max(tx - SAMPLE_RADIUS, 0):min(tx + SAMPLE_RADIUS + 1, self.env_w)
            ]
            mean_conf = np.mean(patch)
            if mean_conf <= lowest_conf:
                lowest_conf = mean_conf
                best_target = (tx, ty)
        
        self.dx = np.sign(best_target[0] - pos_x)
        self.dy = np.sign(best_target[1] - pos_y)
        pos_x += self.dx * SPEED
        pos_y += self.dy * SPEED
        pos_x = max(AGENT_SIZE // 2, min(self.env_w - 1 - AGENT_SIZE // 2, pos_x))
        pos_y = max(AGENT_SIZE // 2, min(self.env_h - 1 - AGENT_SIZE // 2, pos_y))
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
        patch = env_array[y0 : y1 + 1, x0 : x1 + 1, :]

        # Update agent's maps
        self.observed[y0 : y1 + 1, x0 : x1 + 1, :] = patch
        self.explored[y0 : y1 + 1, x0 : x1 + 1] = 1.0
        # Agents are very confident with newly observed areas
        self.confidence[y0 : y1 + 1, x0 : x1 + 1] = 1.0
        if self.policy == "random":
            self.random_walk()
        elif self.policy =="confidence":
            self.confidence_based_walk()

    # Calculates payload based on restriction on payload size in bytes, prioritizing regions with higher confidence
    # Transmission of each pixel costs at least 3 bytes -> uint8 data for 3 channels (RGB)
    def populate_payload(self, max_pixels):
        # When transmissible payload is restricted, prioritize pixels with higher confidence
        # Convert 2D (H, W) into 1D H * W for sorting
        flat_confidence = self.confidence.flatten()
        # Sort in ascending order, get entries at the back, '-max_pixels'
        high_confidence_indices = np.argsort(flat_confidence)[-max_pixels:]

        # unravel_index -> Converts indices in a flattened array back to its unflattened version
        # Reference: https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
        h_indices, w_indices = np.unravel_index(high_confidence_indices, self.confidence.shape)

        observed_payload = self.observed[h_indices, w_indices]

        # print(f"h_indices = {h_indices}")
        # print(f"observed_payload = {observed_payload}")

        # Package containinng list of coordinate pairs ('explored') and the corresponding RGB values
        self.payload = {
            "positions": list(zip(h_indices, w_indices)),
            "observed": observed_payload,
        }
    
    @staticmethod
    # Does not depend on any instance attributes
    def combine_observation(self_obs, incoming_obs):
        if np.all(self_obs == 0.0):
            # If RGB pixel is [0.0, 0.0, 0.0]
            return incoming_obs
        else:
            # For regions that both agents have explored, calculate average
            return (self_obs + incoming_obs) / 2.0

    def communicate_with(self, other_agent, confidence_reception):
        received_payload = other_agent.payload
        if received_payload is None:
            return

        for (h, w), obs_pixel in zip(received_payload["positions"], received_payload["observed"]):
            # print(f"({h}, {w}) → {obs_pixel}")
            self.observed[h, w] = Agent.combine_observation(self.observed[h, w], obs_pixel)
            # Combine explored flags
            self.explored[h, w] = 1.0
            # Assign default confidence to received data
            self.confidence[h, w] = confidence_reception

    '''
    Confidence Mechanism
    - Purpose: Want agents to be able to adapt to dynamic environment
    - Confidence ranges from 0.0 to 1.0
    - Newly observed areas are assigned confidence of 1.0
    - Confidence decays with time
    - When confidence drops below a certain threshold, 'delete' associated observed areas and set explored to False
    '''
    def update_confidence(self):
        # Update confidence matrix
        # self.confidence *= (1 - self.confidence_decay)  # Exponential decay
        self.confidence -= self.confidence_decay # Linear decay
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
        # 'Delete' if below threshold
        low_conf_mask = self.confidence < self.confidence_threshold
        # Update agent's maps
        self.observed[low_conf_mask] = 0.00
        self.explored[low_conf_mask] = 0.00

    def get_position(self):
        return self.position

    def get_observation(self):
        return self.observed

    def get_explored(self):
        return self.explored
    
    def get_confidence(self):
        return self.confidence
    
    def get_payload(self):
        return self.payload
    
    def get_policy(self):
        return self.policy
    
    def set_observation(self, observed):
        self.observed = observed

    def set_explored(self, explored):
        self.explored = explored
    
    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_payload(self, payload):
        self.payload = payload

    # Draw agent, x_offset is provided for the agents to be drawn in the right column of the window
    def draw(self, screen, scale, x_offset, agent_colour):
        pos_x, pos_y = self.position
        pygame.draw.rect(
            screen,
            agent_colour,
            (
                pos_x * scale + x_offset,
                pos_y * scale + 30,
                AGENT_SIZE,
                AGENT_SIZE,
            ),
        )
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (pos_x * scale + x_offset + 7, pos_y * scale + 32),
            2,
        )
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (pos_x * scale + x_offset + 2, pos_y * scale + 32),
            2,
        )
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (pos_x * scale + x_offset + 7, pos_y * scale + 32),
            1,
        )
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (pos_x * scale + x_offset + 2, pos_y * scale + 32),
            1,
        )
