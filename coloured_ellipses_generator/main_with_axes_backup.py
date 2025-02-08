import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_colors = [cmap(i) for i in np.linspace(minval, maxval, n)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors, N=n
    )
    return new_cmap


def anderson_ellipse_ratio(wind_speed_kmph):
    wind_speed_mph = wind_speed_kmph / 1.60934
    val = (
        0.936 * math.exp(0.2566 * wind_speed_mph)
        + 0.461 * math.exp(-0.1548 * wind_speed_mph)
        - 0.397
    )
    return min(val, 8.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single elliptical fire-intensity image."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="dataset/image.png",
        help="Output PNG file path (default=dataset/image.png).",
    )
    parser.add_argument(
        "--width", type=int, default=128, help="Image width in pixels (default=128)."
    )
    parser.add_argument(
        "--height", type=int, default=128, help="Image height in pixels (default=128)."
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=[-5, 5],
        help="x-axis domain (default: -5 5).",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[-5, 5],
        help="y-axis domain (default: -5 5).",
    )
    parser.add_argument(
        "--ignition",
        type=float,
        nargs=2,
        default=None,
        help="Ignition point (x0,y0). Default: center of domain.",
    )
    parser.add_argument(
        "--max_intensity",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Max fire intensity class: 1=low,2=mod,3=high,4=extreme. (default=3)",
    )
    parser.add_argument(
        "--wind", action="store_true", help="Enable wind-based ellipse ratio."
    )
    parser.add_argument(
        "--wind_speed",
        type=float,
        default=10.0,
        help="Wind speed in km/h (default=10). Used if --wind is true.",
    )
    parser.add_argument(
        "--wind_dir",
        type=float,
        default=0.0,
        help="Wind direction in degrees from x-axis (default=0).",
    )
    parser.add_argument(
        "--decay_const",
        type=float,
        default=1.0,
        help="Decay constant k in I(r)=I0*exp(-k*r). (default=1.0)",
    )
    args = parser.parse_args()

    nx, ny = args.width, args.height
    xlim, ylim = args.xlim, args.ylim
    xvals = np.linspace(xlim[0], xlim[1], nx)
    yvals = np.linspace(ylim[0], ylim[1], ny)

    if args.ignition is None:
        x0 = 0.5 * (xlim[0] + xlim[1])
        y0 = 0.5 * (ylim[0] + ylim[1])
    else:
        x0, y0 = args.ignition

    if args.wind:
        ratio = anderson_ellipse_ratio(args.wind_speed)
    else:
        ratio = 1.0

    theta = math.radians(args.wind_dir)
    minor_axis = 3.0
    if ratio < 1.0:
        ratio = 1.0
    major_axis = ratio * minor_axis

    intensity = np.full((ny, nx), np.nan, dtype=np.float32)

    I0 = float(args.max_intensity)
    k = args.decay_const

    cosT = math.cos(theta)
    sinT = math.sin(theta)

    for j in range(ny):
        for i in range(nx):
            dx = xvals[i] - x0
            dy = yvals[j] - y0

            xprime = dx * cosT + dy * sinT
            yprime = -dx * sinT + dy * cosT

            e_dist = (xprime / major_axis) ** 2 + (yprime / minor_axis) ** 2

            if e_dist <= 1.0:
                val = I0 * math.exp(-k * (e_dist))
                intensity[j, i] = val

    fraction = I0 / 4.0
    base_cmap = plt.cm.inferno
    truncated_cmap = truncate_colormap(base_cmap, 0.0, fraction)
    truncated_cmap.set_bad("white")

    outdir = os.path.dirname(args.output)
    if outdir.strip():
        os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        intensity,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        cmap=truncated_cmap,
        vmin=0.0,
        vmax=I0,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Intensity")

    plt.title(f"Ellipse (L/W={ratio:.2f}), max={args.max_intensity}, wind={args.wind}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("image")

    plt.savefig(args.output, dpi="figure")
    plt.close()

    print(f"Saved synthetic fire intensity image => {args.output}")
    print(f"  ratio={ratio:.2f}  minor={minor_axis:.2f}  major={major_axis:.2f}")


if __name__ == "__main__":
    main()
