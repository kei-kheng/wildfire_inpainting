import os
import argparse

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# Only the generator is needed for inference
from models import ContextEncoder
from image_utils import IR_Images, ImageResize, get_transform, apply_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default="models/test2/generator.pth")
    parser.add_argument("--img_scaled_dim", type=int, default=320)  # 1704 x 1280
    parser.add_argument("--coverage", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--num_show", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_transform(args.img_scaled_dim)

    infer_dataset = IR_Images(args.data_dir, ["inference"], transform=transform)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running inference on:", device)

    generator = ContextEncoder(in_channels=3).to(device)
    generator.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    generator.eval()  # Put generator in inference/eval mode
    print(f"Loaded generator from '{args.model_path}'")

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(infer_loader):
            images = images.to(device)  # shape (B,3,H,W)
            masked_imgs, masks = apply_mask(images, coverage=args.coverage)

            g_out = generator(masked_imgs)
            comp = images * masks + g_out * (1 - masks)

            # Visualize and save results
            show_count = min(args.num_show, images.size(0))

            for i in range(show_count):
                # Take the sequence of 3 tensors and concatenate them along the first dimension to create a 3-image row
                out_tensor = torch.stack([images[i], masked_imgs[i], comp[i]], dim=0)  # shape (3,3,H,W)
                save_path = os.path.join(args.output_dir, f"inferred_{filenames[i]}")
                vutils.save_image(out_tensor, save_path, nrow=3, normalize=True)
                print(f"Saved image: {save_path}")

if __name__ == "__main__":
    main()