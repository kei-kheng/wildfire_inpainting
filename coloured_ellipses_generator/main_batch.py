import argparse
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_colors = [cmap(i) for i in np.linspace(minval, maxval, n)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        new_colors,
        N=n,
    )
    return new_cmap


def calc_ellipse_ratio(wind_speed_kmph):
    wind_speed_mph = wind_speed_kmph / 1.60934
    val = (
        0.936 * math.exp(0.2566 * wind_speed_mph)
        + 0.461 * math.exp(-0.1548 * wind_speed_mph)
        - 0.397
    )
    return min(val, 8.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset")
    # Want symmetrical bounds and square images
    parser.add_argument("--no_of_images", type=int, default=10)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--lim", type=float, default=10)
    args = parser.parse_args()

    # Ensure that output folder exists
    os.makedirs(args.output, exist_ok=True)

    nx, ny = args.size, args.size
    x_min, x_max = -args.lim, args.lim
    y_min, y_max = -args.lim, args.lim

    # Constants that are independent of loop
    minor_axis = 3.0
    k = 1.0

    for i in range(args.no_of_images):
        # Any value within the given interval is equally likely to be drawn
        # Reference: https://numpy.org/doc/2.2/reference/random/generated/numpy.random.uniform.html
        # x0 = random.uniform(x_min, x_max)
        # x0 = random.uniform(y_min, y_max)

        # Can uncomment code above for randomized ignition point
        x0, y0 = 0.0, 0.0

        # choice() returns a randomly selected element from the specified sequence
        # Reference: https://www.w3schools.com/python/ref_random_choice.asp
        I0 = random.choice([1, 2, 3, 4])

        # Returns a random number between 0.0 and 1.0, want 50% wind
        do_wind = random.random() < 0.5
        if do_wind:
            wind_dir = random.uniform(0.0, 360.0)
            wind_speed_kmph = random.uniform(0.1, 10.0)
            ratio = calc_ellipse_ratio(wind_speed_kmph)
        else:
            wind_dir = 0.0
            wind_speed_kmph = 0.0
            ratio = 1.0
        ratio = max(1.0, ratio)

        xvals = np.linspace(x_min, x_max, nx)
        yvals = np.linspace(y_min, y_max, ny)

        intensity = np.full((ny, nx), np.nan, dtype=np.float32)

        major_axis = ratio * minor_axis
        theta = math.radians(wind_dir)
        cosT = math.cos(theta)
        sinT = math.sin(theta)

        for row in range(ny):
            for col in range(nx):
                dx = xvals[col] - x0
                dy = yvals[row] - y0
                xprime = dx * cosT + dy * sinT
                yprime = -dx * sinT + dy * cosT

                e_dist = (xprime / major_axis) ** 2 + (yprime / minor_axis) ** 2
                if e_dist <= 1.0:
                    val = I0 * math.exp(-k * e_dist)
                    intensity[row, col] = val

        truncated_cmap = truncate_colormap(plt.cm.inferno, 0.0, I0 / 4.0)
        truncated_cmap.set_bad("white")

        DPI = 200
        fig_w = nx / DPI
        fig_h = ny / DPI
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)

        ax.imshow(
            intensity,
            origin="lower",
            cmap=truncated_cmap,
            vmin=0.0,
            vmax=I0,
            interpolation="bilinear",
        )
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Construct output name
        out_name = f"image_{i:03d}.png"
        out_path = os.path.join(args.output, out_name)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Generated {args.no_of_images} images, stored in {args.output}")


if __name__ == "__main__":
    main()
