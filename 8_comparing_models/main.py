# Libraries
import os
import yaml
import argparse
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models import (
    ConvAutoencoder,
    PartialConvAutoencoder,
    ContextEncoder,
    PatchDiscriminator,
    weights_init,
)
from utils import (
    EllipseDataset,
    ImageResize,
    apply_mask,
    plot_from_csv_training,
    cal_MSE,
    cal_PSNR,
    cal_SSIM,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model_type", type=str, default="CAE")
    parser.add_argument("--output_dir", type=str, default="run1")
    parser.add_argument("--coverage", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.yaml_path:
        with open(args.yaml_path, "r") as f:
            config_args = yaml.safe_load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    os.makedirs(f"results/{args.model_type}/{args.output_dir}/images", exist_ok=True)

    num_show = 5

    # Create/overwrite CSV file
    csv_path = f"results/{args.model_type}/{args.output_dir}/training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "MSE", "PSNR", "SSIM"])

    # Create/overwrite and save training conditions to a text file
    txt_path = f"results/{args.model_type}/{args.output_dir}/training_conditions.txt"
    with open(txt_path, "w") as f:
        for arg_key, arg_value in vars(args).items():
            f.write(f"--{arg_key} {arg_value}\n")
    print(f"Wrote training conditions to: {txt_path}")

    transform = T.Compose(
        [
            ImageResize(scaled_dim=128),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = EllipseDataset(args.data_dir, transform=transform)
    print("Dataset length:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == "CAE":
        model = ConvAutoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif args.model_type == "PCAE":
        model = PartialConvAutoencoder().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif args.model_type == "GAN":
        generator = ContextEncoder(in_channels=3).to(device)
        generator.apply(weights_init)
        discriminator = PatchDiscriminator(in_channels=3).to(device)
        discriminator.apply(weights_init)

        criterion_bce = nn.BCEWithLogitsLoss().to(device)
        criterion_l1 = nn.L1Loss().to(device)
        optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerD = optim.Adam(
            discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        realLabel = 1.0
        fakeLabel = 0.0

    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):  # For each batch
            real_imgs = real_imgs.to(device)
            masked_imgs, masks = apply_mask(real_imgs, coverage=args.coverage)

            if args.model_type in ["CAE", "PCAE"]:
                if args.model_type == "CAE":
                    model_out = model(masked_imgs)
                else:
                    model_out = model(masked_imgs, masks)
                comp = real_imgs * masks + model_out * (1 - masks)
                loss = criterion(comp, real_imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif args.model_type == "GAN":
                discriminator.zero_grad()

                d_out_real = discriminator(real_imgs)
                real_label = torch.full_like(d_out_real, realLabel, device=device)
                lossD_real = criterion_bce(d_out_real, real_label)

                g_out = generator(masked_imgs)
                comp = real_imgs * masks + g_out * (1 - masks)
                d_out_fake = discriminator(comp.detach())
                fake_gt = torch.full_like(d_out_fake, fakeLabel, device=device)
                lossD_fake = criterion_bce(d_out_fake, fake_gt)

                lossD = lossD_real + lossD_fake
                lossD.backward()
                optimizerD.step()

                generator.zero_grad()

                d_out_fakeForG = discriminator(comp)
                adv_gt = torch.full_like(d_out_fakeForG, realLabel, device=device)
                lossG_adv = criterion_bce(d_out_fakeForG, adv_gt)

                lossG_recon = criterion_l1(g_out * (1 - masks), real_imgs * (1 - masks))

                lossG = lossG_adv + 10 * lossG_recon
                lossG.backward()
                optimizerG.step()

            # Calculate MSE, PSNR and SSIM
            MSE_vals = []
            PSNR_vals = []
            SSIM_vals = []

            with torch.no_grad():
                for j in range(real_imgs.size(0)):
                    comp_j = comp[j].cpu().numpy()
                    real_j = real_imgs[j].cpu().numpy()
                    mask_j = masks[j].cpu().numpy()

                    MSE_val = cal_MSE(comp_j, real_j, mask_j[0])
                    PSNR_val = cal_PSNR(comp_j, real_j, mask_j[0])
                    SSIM_val = cal_SSIM(comp_j, real_j, mask_j[0])

                    MSE_vals.append(MSE_val)
                    PSNR_vals.append(PSNR_val)
                    SSIM_vals.append(SSIM_val)

            avg_MSE = np.mean(MSE_vals)
            avg_PSNR = np.mean(PSNR_vals)
            avg_SSIM = np.mean(SSIM_vals)

            print(
                f"Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(dataloader)}] "
                f"MSE = {avg_MSE:.4f}, PSNR = {avg_PSNR:.4f}, SSIM = {avg_SSIM:.4f}"
            )

            # Append epoch number, batch number and losses to created CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                # Epoch / Batch / MSE / PSNR / SSIM
                writer.writerow(
                    [
                        epoch + 1,
                        i + 1,
                        f"{avg_MSE:.4f}",
                        f"{avg_PSNR:.4f}",
                        f"{avg_SSIM:.4f}",
                    ]
                )

        # Visualize and save results from last batch of every epoch
        with torch.no_grad():
            sample_real = real_imgs[:num_show].cpu()
            sample_masked = masked_imgs[:num_show].cpu()
            sample_comp = comp[:num_show].cpu()

            row = torch.cat(
                [sample_real[i].unsqueeze(0) for i in range(num_show)]
                + [sample_masked[i].unsqueeze(0) for i in range(num_show)]
                + [sample_comp[i].unsqueeze(0) for i in range(num_show)],
                dim=0,
            )

            img = vutils.make_grid(
                row.view(-1, *sample_real.shape[1:]), nrow=num_show, normalize=True
            )
            vutils.save_image(
                img,
                f"results/{args.model_type}/{args.output_dir}/images/epoch_{epoch+1}.png",
            )

    # Plot graphs from CSV
    plot_from_csv_training(args.model_type, args.output_dir)
    print("Plotted graphs from CSV")


if __name__ == "__main__":
    main()
