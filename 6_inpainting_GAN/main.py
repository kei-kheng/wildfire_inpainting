# Libraries
import os
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim
# Contains various utilities, mostly for visualization
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from models import ContextEncoder, PatchDiscriminator
from image_utils import IR_Images, get_transform, apply_mask, plot_from_csv

# Weight initialization according to layer
# Reference: DCGAN Radford et al., 2015, + Ioffe & Szegedy
def weights_init(layer):
    classname = layer.__class__.__name__
    if "Conv" in classname or "ConvTranspose" in classname:  # Mean = 0, Std = 0.02
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/IR_images")
    parser.add_argument("--folders", nargs="+", default=["5"])
    parser.add_argument("--img_scaled_dim", type=int, default=320)  # 1704 x 1280
    parser.add_argument("--coverage", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_recon", type=int, default=10)
    # Default values based on best practice - learning rate, beta
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="default_folder")
    parser.add_argument("--num_show", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(f"results/{args.output_dir}/real", exist_ok=True)
    os.makedirs(f"results/{args.output_dir}/masked", exist_ok=True)
    os.makedirs(f"results/{args.output_dir}/recon", exist_ok=True)
    os.makedirs(f"models/{args.output_dir}", exist_ok=True)

    # Create/overwrite CSV file
    # 'newline' argument is needed to avoid extra blank lines/newline issues: https://docs.python.org/3/library/csv.html#csv.writer
    csv_path = f"results/{args.output_dir}/training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "LossD", "LossG", "LossG_recon"])

    transform = get_transform(args.img_scaled_dim)
    
    dataset = IR_Images(args.data_dir, args.folders, transform=transform)  # IR_Images loads dataset in a sorted order
    print(f"Found {len(dataset)} images across subfolders {args.folders}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # Shuffle batches during each epoch

    # Create/overwrite and save training conditions to a text file
    # vars() returns a dict, items() returns an iterable of (key, value) pairs
    txt_path = f"models/{args.output_dir}/training_conditions.txt"
    with open(txt_path, "w") as f:
        for arg_key, arg_value in vars(args).items():
            f.write(f"--{arg_key} {arg_value}\n")
        f.write(f"Dataset length: {len(dataset)}\n")
    print(f"Wrote training conditions to: {txt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    generator = ContextEncoder(in_channels=3).to(device)
    # Custom initialization of neural network layer weights
    generator.apply(weights_init)

    discriminator = PatchDiscriminator(in_channels=3).to(device)
    discriminator.apply(weights_init)

    # Criteria/Losses
    # For PatchDiscriminator. The logarithms are handled inside this loss function -> No explicit log calls in the code.
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    # For reconstruction, https://outcomeschool.com/blog/l1-and-l2-loss-functions
    criterion_l1 = nn.L1Loss().to(device)

    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Labelling
    realLabel = 1.0
    fakeLabel = 0.0

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):  # For each batch
            real_imgs = real_imgs.to(device)
            masked_imgs, masks = apply_mask(real_imgs, coverage=args.coverage)

            # ------ Train discriminator ------
            """
            Previously, zero_grad() was called on the Adam optimizer. In most cases, calling it on the model has the 
            same effect (in most simple use cases) as calling it on the optimizer, with some subtle differences:
            - optimizer.zero_grad() only zeroes out the gradients of the parameters that that optimizer is responsible for.
            - model.zero_grad() loops through every parameter in the model and zeros the .grad field.
            """
            discriminator.zero_grad()

            # Shape is (B,1,H/8,W/8) spatial map
            d_out_real = discriminator(real_imgs)
            # Create a tensor of 1s (real_label), same size as d_out_real
            real_label = torch.full_like(d_out_real, realLabel, device=device)
            # criterion_bce applies signmoid to d_out_real before taking logarithm
            lossD_real = criterion_bce(d_out_real, real_label)

            # Generator's output is in the [-1, 1] range due to tanh()
            g_out = generator(masked_imgs)
            # Composite: Retain original pixels in the known pixels (real_img * masks), add generated pixels in the unknown region (g_out * (1 - masks))
            comp = real_imgs * masks + g_out * (1 - masks)
            # See if discriminator could detect that the images are fake (generated)
            # detach() to prevent upating generator's parameters -> See logbook to see usage of detach()
            d_out_fake = discriminator(comp.detach())
            fake_gt = torch.full_like(d_out_fake, fakeLabel, device=device)
            lossD_fake = criterion_bce(d_out_fake, fake_gt)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # ------ Train generator ------
            generator.zero_grad()

            # Adversarial loss: Want discriminator(comp) => real
            d_out_fakeForG = discriminator(comp)
            adv_gt = torch.full_like(d_out_fakeForG, realLabel, device=device)
            # How far away from 'real' are the generated images?
            lossG_adv = criterion_bce(d_out_fakeForG, adv_gt)

            # Calculate reconstruction loss in the missing (unknown) regions only
            lossG_recon = criterion_l1(g_out * (1 - masks), real_imgs * (1 - masks))

            """
            Generator loss
            - Combines adversarial loss and more 'objective' reconstruction loss
            - Larger weight for latter (lambda_recon), want model to preserve known data
            - References: Isola et al., 2017 + Pathak et al., 2016 + pix2pix
            """
            lossG = lossG_adv + args.lambda_recon * lossG_recon
            lossG.backward()
            optimizerG.step()

            # PyTorch item() returns a scalar value from a single-element tensor - different from items()!
            print(
            f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(dataloader)}] "
            f"LossD: {lossD.item():.4f}, LossG: {lossG.item():.4f}, LossG_recon: {lossG_recon.item():.4f}"
            )

            # Append epoch number, batch number and losses to created CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                # Epoch / Batch / LossD / LossG / LossG_recon
                writer.writerow([epoch + 1, i + 1, f"{lossD.item():.5f}", f"{lossG.item():.5f}", f"{lossG_recon.item():.5f}"])

        # Visualize and save results from last batch of every epoch
        with torch.no_grad():
            sample_real = real_imgs[: args.num_show].cpu()
            sample_masked = masked_imgs[: args.num_show].cpu()
            sample_comp = comp[: args.num_show].cpu()

        # A more convenient way of visualization compared to matplotlib
        vutils.save_image(
            sample_real, f"results/{args.output_dir}/real/epoch_{epoch+1}.png", normalize=True
        )
        vutils.save_image(
            sample_masked, f"results/{args.output_dir}/masked/epoch_{epoch+1}.png", normalize=True
        )
        vutils.save_image(
            sample_comp, f"results/{args.output_dir}/recon/epoch_{epoch+1}.png", normalize=True
        )

    # Save models - discriminator is usally not needed once training is complete
    torch.save(generator.state_dict(), f"models/{args.output_dir}/generator.pth")
    print(f"Saved Context Encoder to: models/{args.output_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"models/{args.output_dir}/discriminator.pth")
    print(f"Saved Patch Discriminator to: models/{args.output_dir}/discriminator.pth")

    # Plot graphs from CSV
    plot_from_csv(args.output_dir)
    print("Plotted graphs from CSV")

if __name__ == "__main__":
    main()