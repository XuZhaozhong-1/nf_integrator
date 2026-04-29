import torch
import time

from nf.losses import kl_loss, raw_std_loss, kl_raw_std_loss


def train_nf(
    model,
    benchmark_fn,
    dim,
    steps=2000,
    batch_size=4096,
    lr=1e-3,
    loss_name="kl",
    device="cpu",
    eta=0.5,
    verbose=True,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    t0 = time.time()

    for step in range(steps):
        x = torch.rand(batch_size, dim, device=device)
        f_vals = benchmark_fn(x)
        f_vals = f_vals / (f_vals.mean() + 1e-12)
        log_q = model.log_prob(x)

        if loss_name == "kl":
            loss = kl_loss(log_q, f_vals)

        elif loss_name == "raw_std_loss":
            loss = raw_std_loss(log_q, f_vals)

        elif loss_name == "kl_raw_std_loss":
            # lambda goes from 1 -> 0 across training
            loss = kl_raw_std_loss(log_q, f_vals,eta=eta)

        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        history.append(float(loss.item()))

        if verbose and step % 200 == 0:
            print(f"step {step:4d} | loss = {loss.item():.6f}")

    runtime = time.time() - t0

    return {
        "model": model,
        "history": history,
        "runtime": runtime,
        "loss_name": loss_name,
        "dim": dim,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "eta": eta,
    }


def save_nf_checkpoint(train_out, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": train_out["model"].state_dict(),
            "history": train_out["history"],
            "runtime": train_out["runtime"],
            "loss_name": train_out["loss_name"],
            "dim": train_out["dim"],
            "steps": train_out["steps"],
            "batch_size": train_out["batch_size"],
            "lr": train_out["lr"],
            "eta": train_out["eta"],
        },
        path,
    )

    print(f"Saved checkpoint -> {path}")