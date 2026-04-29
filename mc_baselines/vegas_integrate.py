import time
import torch
import vegas


def vegas_integrate(f, dim, nitn, neval, device="cpu", dtype=torch.float32):
    """
    Integrate f over [0,1]^dim with VEGAS.

    Parameters
    ----------
    f : callable
        Function taking a torch tensor of shape (N, dim) and returning shape (N,).
    dim : int
        Integration dimension.
    nitn : int
        Number of VEGAS iterations.
    neval : int
        Number of evaluations per iteration.
    device : str
        Torch device for benchmark evaluation.
    dtype : torch.dtype
        Torch dtype for benchmark evaluation.

    Returns
    -------
    dict
        {
            "estimate": float,
            "variance": float,
            "runtime": float,
            "events": int,
            "nitn": int,
            "neval": int
        }
    """
    integ = vegas.Integrator(dim * [[0.0, 1.0]])

    def f_vegas(x):
        # VEGAS passes one point x with shape (dim,)
        xt = torch.tensor([x], dtype=dtype, device=device)
        y = f(xt)
        return float(y[0].item())

    t0 = time.time()
    result = integ(f_vegas, nitn=nitn, neval=neval)
    runtime = time.time() - t0

    return {
        "estimate": float(result.mean),
        "variance": float(result.sdev ** 2),
        "runtime": float(runtime),
        "events": int(nitn * neval),
        "nitn": int(nitn),
        "neval": int(neval),
    }