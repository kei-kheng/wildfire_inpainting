import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Create a new colourmap that is a subset of the original
# Reference: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_colors = [cmap(i) for i in np.linspace(minval, maxval, n)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        new_colors,
        N=n,  # N=n sets discrete levels used to sample colourmap
    )
    return new_cmap


# Compute ellipse length to width ratio (L/W) based on Anderson (1983)
# Reference: https://elmfire.io/verification/verification_01.html
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
    parser.add_argument("--output", default="dataset/image.png")
    # Want width=height and xlim=ylim for 1:1 aspect ratio
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--lim", type=float, nargs=2, default=[-10, 10])
    parser.add_argument("--ignition_point", type=float, nargs=2, default=None)
    parser.add_argument("--max_intensity", type=int, default=3, choices=[1, 2, 3, 4])
    # Create a default value of True when command-line argument is present
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind_speed", type=float, default=5.0)
    parser.add_argument("--wind_dir", type=float, default=0.0)
    args = parser.parse_args()

    # Define domain, lim is a 2-element list
    nx, ny = args.size, args.size
    xlim, ylim = args.lim, args.lim
    xvals = np.linspace(xlim[0], xlim[1], nx)
    yvals = np.linspace(ylim[0], ylim[1], ny)

    # Set ignition point to center by default
    if args.ignition_point is None:
        x0 = 0.5 * (xlim[0] + xlim[1])
        y0 = 0.5 * (ylim[0] + ylim[1])
    else:
        x0, y0 = args.ignition_point

    # Calculate ellipse ratio (set to 1 when there is no wind)
    if args.wind:
        ratio = calc_ellipse_ratio(args.wind_speed)
    else:
        ratio = 1.0
    ratio = max(1.0, ratio)

    theta = math.radians(args.wind_dir)
    # Width of ellipse, reference length (could be set to any arbitrary value)
    minor_axis = 3.0
    major_axis = ratio * minor_axis  # Length of ellipse

    # Create 2D NumPy array of size height x width, fill with np.nan (no data)
    # np.nan written explicitly so that the background can be assigned a colour that is not in the cmap
    intensity = np.full((ny, nx), np.nan, dtype=np.float32)

    # Initial fire intensity (cap/maximum)
    I0 = float(args.max_intensity)
    # Decay factor (based on fuel and environmental factors), set to 1 for a simplified fire model
    k = 1.0

    # Used for rotation
    cosT = math.cos(theta)
    sinT = math.sin(theta)

    # Colourize ellipse
    for j in range(ny):
        for i in range(nx):
            # Offset/translate each point accordingly according to the ignition point
            dx = xvals[i] - x0
            dy = yvals[j] - y0
            # Rotate by -theta
            xprime = dx * cosT + dy * sinT
            yprime = -dx * sinT + dy * cosT
            # Equation of an ellipse: https://www.cuemath.com/geometry/ellipse/, how far from center
            e_dist = (xprime / major_axis) ** 2 + (yprime / minor_axis) ** 2
            # If within boundary of ellipse
            if e_dist <= 1.0:
                # Define intensity as a decayinig exponential dependent on distance from center of ellipse
                val = I0 * math.exp(-k * e_dist)
                intensity[j, i] = val

    # Truncate cmap according to the specified maximum fire intensity
    # inferno's clour spectrum is closest to to IR image's, '_r' suffix indicates the 'reversed' version
    truncated_cmap = truncate_colormap(plt.cm.inferno, 0.0, I0 / 4.0)
    # Set colour for masked values (NaN) to white (background)
    truncated_cmap.set_bad("white")

    outdir = os.path.dirname(args.output)
    if outdir.strip():
        os.makedirs(outdir, exist_ok=True)

    # Enforce a size of 'size x size' pixels
    # Division by DPI is necessary (see logbook for explanation)
    DPI = 200
    fig_w = nx / DPI
    fig_h = ny / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)

    # Render intensity as a 2D image on the 'Axes' object, 'ax': https://matplotlib.org/stable/api/axes_api.html
    ax.imshow(
        intensity,
        origin="lower",  # Places array row 0 at the bottom of plot (standard Cartesian). Else, row 0 would be at the top.
        cmap=truncated_cmap,
        vmin=0.0,
        vmax=I0,  # Colour scale range (can be displayed as the 'Intensity' bar)
        interpolation="bilinear",  # Smoothens image (this does improve the quality of the image)
    )

    # Remove axis lines, ticks and labels
    ax.set_axis_off()

    # subplots_adjust uses normalized coordiantes within the range [0..1]
    # Place left boundary at extreme left, place right boundary at extreme right
    # Similar for top and bottom. Ensures no horizontal and vertical margin
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Save to args.output with minimised surrounding whitespace and 0 padding
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0)
    # Close figure to save memory
    plt.close(fig)

    # Debug outputs
    print(f"Saved synthetic fire intensity to '{args.output}' ({nx}x{ny}px)")


if __name__ == "__main__":
    main()
