from functions import double_gaussian, slashed_circle, gaussian_family
from plot import plot_2d_function
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
plot_2d_function(double_gaussian, title="double gaussian")

plot_2d_function(slashed_circle, title="slashed circle")

n = 200
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z_slices = [0.25, 0.5, 0.75]

fig, axes = plt.subplots(1, len(z_slices), figsize=(15, 4))

all_values = []

# first pass to get global min/max
for z0 in z_slices:
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full_like(X, z0)

    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    grid_torch = torch.tensor(grid, dtype=torch.float32)

    values = gaussian_family(grid_torch).detach().cpu().numpy().reshape(n, n)
    all_values.append(values)

vmin = min(v.min() for v in all_values)
vmax = max(v.max() for v in all_values)

# plotting
for ax, z0, values in zip(axes, z_slices, all_values):

    values_plot = np.log(values + 1e-12)

    cf = ax.contourf(
        X, Y, values_plot,
        levels=40,
        cmap="viridis"
    )

    ax.contour(
        X, Y, values_plot,
        levels=10,
        colors="white",
        linewidths=0.6,
        alpha=0.7
    )

    ax.set_title(f"x3 = {z0:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")

fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.8, label="log gaussian_family(x)")

plt.savefig("plots/gaussian_family_slices_improved.png", dpi=300, bbox_inches="tight")
plt.close()