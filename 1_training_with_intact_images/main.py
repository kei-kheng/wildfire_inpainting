"""
Training Pipeline:
1. Parse command-line args (--train_dir, --epochs, --batch_size, --model_out).
2. Split the dataset into 90% training, 10% testing.
3. Use a DataLoader to feed batches of images to the model: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html.
4. Forward pass: The model outputs reconstructed images.
5. Compute loss against the original images (mean squared error).
6. Backward pass: Update model weights using Adam optimizer.
7. Evaluate on the test set and print training/test losses per epoch.
8. Save the model to a .pth file.
9. Visualize a few reconstructions (of intact images) from the test set.
"""

import os  # Want to use functions to interact with the OS, e.g., file paths
import glob  # Short for 'global', searches for files that match a specific pattern/name
import argparse  # Enables parsing comman-line arguments
from PIL import Image  # For opening and manipulating images in Python
import torch  # PyTorch
import numpy as np  # To deal with numerical operations
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T  # Image transformations
import torch.nn as nn  # Neural network
import torch.nn.functional as F  # Provides functinal versions of many layers and activation functions
import torch.optim as optim  # Houses optimisation functions
import matplotlib.pyplot as plt  # A plotting library

# A class that inherits from Dataset (PyTorch base class)
class EllipseDataset(Dataset):
    """
    Methods with double underscores are special/'magic' methods in Python, allows us to tap into Python's built-in behaviours
    __init__: Constructor, 'files' is a list of sorted string paths to images (with the '.png' extension) in the specified directory (data_dir). 'transform' is the transform function applied to each image.
    __getitem_: Called when you do dataset[idx]
    """

    def __init__(self, data_dir, transform=None):
        # Gather all .png in that directory
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # Loads image as a PIL object and transforms it to grayscale
        img = Image.open(img_path).convert("L")
        if self.transform:  # If there is a transform function
            img = self.transform(img)
        return img

# A class that inherits from PyTorch's nn.Module
# This is how custom neural networks are built in PyTorch
class ConvAutoencoder(nn.Module):
    def __init__(self):
        # super() used to give access to methods and properties of a parent/sibling class
        super(ConvAutoencoder, self).__init__()
        # Encoder
        """
        Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html, good for learning spatial features of images
        Example: self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        - Apply 2D convolution on 1-channel image (grayscale)
        - Produces 32 feature maps
        - 3x3 kernel size, this is the size of the filter that slides over the input image
        - 'padding = 1' means adding a 1-pixel-wide border around input -> Preserves spatial dimension of input image after convolution (128x128 for given input)
            - Without padding, the ouput dimension shrinks: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        """

        """
        -------
        Strides
        -------
        - How many pixels the filter moves each time it slides over the input
        - Different effects!
            - Convolution layer: Stride of 2 means halving the spatial resolution
            - Transposed convolution: Stride of 2 means doubling the spatial resolution
        """
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pooling layers are used to reduce the spatial dimension. stride = 2
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Upsampling
        self.dec2 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    """
    Channel Dimension vs. Spatial Dimension
    - In the encoder, the spatial dimension of the image is reduced using the pooling layers
    - Channel capacity is increased so network can capture more abstract or deeper features
    TAKEAWAY: As you go deeper, you can learn more complex features hence the greater number of channels
    - In the decoder, it goes from a higher-dimensional feature representation down to a single grayscale channel
    """

    def forward(self, x):
        x = F.relu(self.enc1(x))  # ReLU: Activation function to introduce non-linearity
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.dec1(x))
        # Sigmoid: Outputs pixel values in the 0-1 range, convenient for grayscale images
        x = torch.sigmoid(self.dec2(x))
        return x

def main():
    # Read command-line arguments: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()  # Instantiate
    parser.add_argument(
        "--train_dir",
        type=str,
        default="dataset/training",
        help="Path to folder of training images.",  # Printed when -h or --help is supplied at command line (-short, --long), example usage: --train_dir help
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--model_out",
        type=str,
        default="model/autoencoder.pth",
        help="Where to save the trained model.",
    )
    args = parser.parse_args()  # Contains the extracted data from command line

    # Directory containing images for training
    # Conversion to absolute path
    train_dir = os.path.abspath(args.train_dir)
    print("Training data directory:", train_dir)

    # Define transform to preprocess the training images
    # T.Compose() takes a lists of transformations and chains them together so they are applied in sequence to each image
    transform = T.Compose(
        [
            T.Resize((128, 128)),
            # T.RandomRotation(20),  # Uncomment to avoid overfitting
            # T.RandomResizedCrop((128,128), scale=(0.8,1.0)),
            # Conversion from grayscale PIL image to PyTorch tensor (tensor: multi-dimensional array of data that can be used to represent and manipulate data)
            T.ToTensor(),
        ]
    )

    # Instantiate, providing the path containing the training dataset
    dataset = EllipseDataset(train_dir, transform=transform)
    print("Dataset length:", len(dataset))

    # Split data into training set (90%) and testing set (10%, never used for training)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Shuffle data used for training, don't shuffle data used for testing. These are iterators for the mini-batches.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Checks if GPU support CUDA - Compute Unified Device Architecture, programming model that leverages GPUs for speed ups
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Instantiates autoencoder, trains it on 'device'
    model = ConvAutoencoder().to(device)

    # Mean squared error
    criterion = nn.MSELoss()
    # Adam optimiser with learning rate 0.001
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = args.epochs

    # Training loop
    for epoch in range(num_epochs):
        # Tells PyTorch we are in training mode, affects Dropout and BatchNorm which behave differently in training and evaluation phase
        model.train()
        running_loss = 0.0

        # Each iteration yields a mini-batch of 'batch-size' (default = 8) images
        for images in train_loader:
            # Move the 'images' tensor to wherever the model is being trained on (avoid devuce mismatch error)
            images = images.to(device)  # shape (B,1,128,128)

            # Implicitly calls the model's 'forward', that's how nn.Module is designed
            # https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460 - See reply from ptrblck
            # https://www.geeksforgeeks.org/__call__-in-python/ - __call__ in Python
            outputs = model(images)
            # Compare outputs and images to calculate MSE
            loss = criterion(outputs, images)

            # Zeroes out all model parameters before computing next backprop step (don't want gradient from previous batches)
            optimizer.zero_grad()
            # Back propagation
            loss.backward()
            # Adjust weights
            optimizer.step()
            # loss.item() returns scalar float from loss tensor, images.size(0) returns batch size
            running_loss += loss.item() * images.size(0)

        # Average MSE per image over entire training set for a given epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        # Avoids tracking gradients (for backpropagation) in this phase
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
        test_loss = test_loss / len(test_loader.dataset)

        # Formatted strings, hence the 'f', expressions and variables within {}
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # Save model
    out_path = os.path.abspath(args.model_out)
    # Create folder if it does not exist yet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to '{out_path}'")

    # Quick visualisation from one batch from the test set
    model.eval()
    # test_loader is an iterator
    sample_batch = next(iter(test_loader))
    sample_imgs = sample_batch.to(device)
    with torch.no_grad():
        # recon is the hopefully reconstructed outputs from the model
        recon = model(sample_imgs)

    # Move data back to CPU and convert PyTorch tensors to NumPy arrays so that they can be fed into matplotlib for plotting
    sample_imgs = sample_imgs.cpu().numpy()
    recon = recon.cpu().numpy()

    # Visualise at least 5 images or the smaller number if the batch size < 5
    num_show = min(5, sample_imgs.shape[0])
    plt.figure(figsize=(10, 4))
    for i in range(num_show):
        # subplot(total_no_of_rows, total_no_of_columns, pick (i+1)th cell)
        ax1 = plt.subplot(2, num_show, i + 1)
        plt.imshow(sample_imgs[i, 0], cmap="gray")  # ith image, channel 0 (grayscale)
        plt.axis("off")  # Hide axis for cleaner view
        ax2 = plt.subplot(2, num_show, i + 1 + num_show)
        plt.imshow(recon[i, 0], cmap="gray")
        plt.axis("off")
    plt.show()  # Display figure

# Common Python idiom meaning: "Only execute the following code if this script is run as the main program, and not if it’s imported as a module."
# e.g., __name__ will be set to __main__ if you run python main.py, and module name if you import it using 'import main'
if __name__ == "__main__":
    main()