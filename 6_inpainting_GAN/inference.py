import os
import argparse
import csv

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# Only the generator is needed for inference
from models import ContextEncoder
from image_utils import (
    IR_Images, 
    get_transform, 
    apply_mask,
    plot_from_csv_inferencing,
    cal_MSE,
    cal_PSNR,
    cal_SSIM
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/inference")  
    parser.add_argument("--folders", nargs="+", default=["3"]) 
    parser.add_argument("--model_path", type=str, default="models/test/generator.pth")
    parser.add_argument("--img_scaled_dim", type=int, default=320)  # 1704 x 1280
    parser.add_argument("--coverage", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--num_show", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(f"inference_results/{args.output_dir}", exist_ok=True)

    transform = get_transform(args.img_scaled_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running inference on:", device)

    # Load model
    generator = ContextEncoder(in_channels=3).to(device)
    generator.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    generator.eval()  # Put generator in inference/eval mode
    print(f"Loaded generator from '{args.model_path}'")

    # Write to CSV
    csv_path = f"inference_results/{args.output_dir}/inference_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Folder", "Image", "MSE", "PSNR", "SSIM"])

    # Load each folder separately to write to CSV
    for folder in args.folders:
        # Load inference dataset
        infer_dataset = IR_Images(args.data_dir, [folder], transform=transform)
        infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Found {len(infer_dataset)} images in folder '{folder}'")

        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(infer_loader):
                images = images.to(device)  # shape (B,3,H,W)
                masked_imgs, masks = apply_mask(images, coverage=args.coverage)
                g_out = generator(masked_imgs)
                comp = images * masks + g_out * (1 - masks)
                
                # Debug
                # print("comp shape:", comp.shape)
                # print("images shape:", images.shape)
                # print("masks shape:", masks.shape)
                # print("g_out shape:", g_out.shape)
                
                # print("comp:", comp)
                # print("images:", images)
                # print("masks:", masks)
                # print("g_out:", g_out)'

                # Calculate MSE, PSNR and SSIM
                for i in range(images.size(0)):
                    comp_i = comp[i].cpu().numpy()
                    real_i = images[i].cpu().numpy()
                    mask_i = masks[i].cpu().numpy()

                    MSE_val = cal_MSE(comp_i, real_i, mask_i[0])
                    PSNR_val = cal_PSNR(comp_i, real_i, mask_i[0])
                    SSIM_val = cal_SSIM(comp_i, real_i, mask_i[0])

                    # os.path.splitext(): Split into root and extension
                    fname = filenames[i]
                    fname = os.path.splitext(fname)[0]

                    # Write a row
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # Folder / Image / MSE / PSNR / SSIM
                        writer.writerow([
                            folder, 
                            fname,
                            f"{MSE_val:.4f}",
                            f"{PSNR_val:.4f}", 
                            f"{SSIM_val:.4f}"
                        ])
                
                # Visualize and save results
                show_count = min(args.num_show, images.size(0))

                for i in range(show_count):
                    # Take the sequence of 3 tensors and concatenate them along the first dimension to create a 3-image row
                    out_tensor = torch.stack([images[i], masked_imgs[i], comp[i]], dim=0)  # shape (3,3,H,W)
                    save_path = os.path.join(f"inference_results/{args.output_dir}", f"inferred_{filenames[i]}")
                    vutils.save_image(out_tensor, save_path, nrow=3, normalize=True)
                    print(f"Saved image: {save_path}")
    
    plot_from_csv_inferencing(args.output_dir)
    print("Plotted graphs from CSV")

if __name__ == "__main__":
    main()