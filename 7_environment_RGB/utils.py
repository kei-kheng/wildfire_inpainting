import pygame
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Scales image according to provided 'img_scaled_dim'
class ImageResize:
    def __init__(self, scaled_dim, multiple):
        self.scaled_dim = scaled_dim
        self.multiple = multiple

    def __call__(self, img):
        w, h = img.size

        if w >= h:
            new_w = self.scaled_dim
            new_h = int(h * (self.scaled_dim / float(w)))
        else:
            new_h = self.scaled_dim
            new_w = int(w * (self.scaled_dim / float(h)))

        snapped_w = max(self.multiple, (new_w // self.multiple) * self.multiple)
        snapped_h = max(self.multiple, (new_h // self.multiple) * self.multiple)

        return T.Resize((snapped_h, snapped_w))(img)


def img_to_nparray(img_path, scaled_dim, multiple=16):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([ImageResize(scaled_dim=scaled_dim, multiple=multiple)])
    return np.array(transform(img))


def img_to_tensor(img_path, scaled_dim, multiple=16):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose(
        [
            ImageResize(scaled_dim=scaled_dim, multiple=multiple),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(img)

# Convert NumPy array to Pygame surface
def nparray_to_surface(nparray, scale, grayscale=False):
    if grayscale:
        # Stack thrice, Pygame expects 3 channels
        nparray = np.stack((nparray, nparray, nparray), axis=-1)
    array_scaled = np.kron(nparray, np.ones((scale, scale, 1)))

    # NumPy -> (H, W, C) to  Pygame -> (W, H, 3)
    surface = pygame.surfarray.make_surface(np.transpose(array_scaled, (1, 0, 2)))
    return surface