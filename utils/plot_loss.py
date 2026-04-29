import os
import matplotlib.pyplot as plt


def plot_loss_curve(history, save_path, title="Training loss"):
    plt.figure()
    plt.plot(range(len(history)), history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved loss plot -> {save_path}")