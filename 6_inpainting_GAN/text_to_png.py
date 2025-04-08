import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def truncate_colormap(cmap, minval=0, maxval=1.0, n=256):
    new_colors = [cmap(i) for i in np.linspace(minval, maxval, n)]
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        new_colors,
        N=n,
    )
    return new_cmap

dir = "dataset/inference/james/text"
txt_file_name = "1"
txt_file_path = f"{dir}/{txt_file_name}.txt"
txt_file = np.loadtxt(txt_file_path)
out_dir = f"dataset/inference/james/images/{txt_file_name}.png"

truncated_cmap = truncate_colormap(plt.cm.inferno, 0.2, 1)

plt.imshow(txt_file, cmap=truncated_cmap)
plt.axis("off")
# plt.colorbar()
plt.savefig(out_dir, dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()
plt.close()