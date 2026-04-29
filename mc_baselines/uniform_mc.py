import torch
import time

def uniform_mc_integrate(f, dim, N):
    t0 = time.time()

    x = torch.rand(N, dim)

    fx = f(x)

    I_hat = fx.mean()

    var = fx.var(unbiased=True) / N

    runtime = time.time() - t0

    return {
        "estimate": I_hat.item(),
        "variance": var.item(),
        "runtime": runtime
    }