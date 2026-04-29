import torch
import matplotlib.pyplot as plt
import os

def plot_2d_function(f, n=200, title="function", save_dir="plots"):
    """
    f: function (N,2) -> (N,)
    """

    os.makedirs(save_dir, exist_ok=True)

    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)

    xx, yy = torch.meshgrid(x, y, indexing="ij")

    pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    zz = f(pts).reshape(n, n).detach()

    plt.figure(figsize=(6,5))
    plt.contourf(xx.numpy(), yy.numpy(), zz.numpy(), levels=50)
    plt.colorbar()
    plt.title(title)

    filename = f"{save_dir}/{title.replace(' ','_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved plot → {filename}")