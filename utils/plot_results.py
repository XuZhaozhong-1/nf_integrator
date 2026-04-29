import json
import os
import matplotlib.pyplot as plt


def plot_compare(
    json_paths,
    labels,
    metric,
    save_path,
    title=None,
    ylabel=None,
    logy=False,
):
    plt.figure()

    for path, label in zip(json_paths, labels):
        with open(path, "r") as f:
            data = json.load(f)

        events = [d["events"] for d in data]
        values = [d[metric] for d in data]

        plt.plot(events, values, marker="o", label=label)

    plt.xlabel("Number of events")
    plt.ylabel(ylabel if ylabel is not None else metric)
    plt.title(title if title is not None else f"{metric} comparison")
    plt.grid()
    plt.legend()

    if logy:
        plt.yscale("log")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot -> {save_path}")