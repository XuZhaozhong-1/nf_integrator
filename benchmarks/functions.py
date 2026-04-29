import torch


def double_gaussian(x):
    """
    x: tensor shape (N,2)
    domain: [0,1]^2
    """

    mu1 = torch.tensor([0.25, 0.25], device=x.device)
    mu2 = torch.tensor([0.75, 0.75], device=x.device)

    sigma = 0.08

    r1 = ((x - mu1) ** 2).sum(dim=1)
    r2 = ((x - mu2) ** 2).sum(dim=1)

    return torch.exp(-r1 / sigma**2) + torch.exp(-r2 / sigma**2)

def slashed_circle(x):
    """
    x: (N,2)
    """

    cx = 0.5
    cy = 0.5

    dx = x[:,0] - cx
    dy = x[:,1] - cy

    r = torch.sqrt(dx**2 + dy**2)

    ring = torch.exp(-(r - 0.3)**2 / 0.05**2)

    slash = torch.exp(-(x[:,0] - x[:,1])**2 / 0.02**2)

    return torch.clamp(ring + slash, max=1.0)
    
def rotated_precision_2d(lam1, lam2, theta, device, dtype):
    c = torch.cos(torch.tensor(theta, device=device, dtype=dtype))
    s = torch.sin(torch.tensor(theta, device=device, dtype=dtype))
    R = torch.tensor([[c, -s],
                      [s,  c]], device=device, dtype=dtype)
    D = torch.diag(torch.tensor([lam1, lam2], device=device, dtype=dtype))
    return R @ D @ R.T


def gaussian_family(x):
    """
    Multi-peak Gaussian benchmark in arbitrary dimension with
    clearly different orientations in the first 2 coordinates.

    x: (N, d)
    returns: (N,)
    """
    N, d = x.shape
    device = x.device
    dtype = x.dtype

    # centers
    c1 = torch.linspace(0.15, 0.30, d, device=device, dtype=dtype)
    c2 = torch.linspace(0.80, 0.65, d, device=device, dtype=dtype)
    c3 = torch.cat([
        torch.full((d // 2,), 0.20, device=device, dtype=dtype),
        torch.full((d - d // 2,), 0.78, device=device, dtype=dtype),
    ])
    c4 = torch.cat([
        torch.full((d // 3,), 0.75, device=device, dtype=dtype),
        torch.full((d // 3,), 0.25, device=device, dtype=dtype),
        torch.full((d - 2 * (d // 3),), 0.60, device=device, dtype=dtype),
    ])

    centers = [c1, c2, c3, c4]
    amps = [1.0, 0.8, 1.4, 0.6]

    # Strongly different xy orientations
    # + diagonal, - diagonal, near horizontal, near vertical
    Axy_list = [
        rotated_precision_2d(0.35, 7.0,  0.80, device, dtype),
        rotated_precision_2d(0.35, 7.0, -0.80, device, dtype),
        rotated_precision_2d(0.25, 8.0,  0.15, device, dtype),
        rotated_precision_2d(0.25, 8.0,  1.35, device, dtype),
    ]

    # Optional extra rotated block in coords (3,4) if d >= 4
    A34_list = [
        rotated_precision_2d(0.8, 3.5,  0.30, device, dtype),
        rotated_precision_2d(0.8, 3.5, -0.50, device, dtype),
        rotated_precision_2d(0.6, 4.0,  0.90, device, dtype),
        rotated_precision_2d(0.6, 4.0, -1.00, device, dtype),
    ]

    # diagonal precisions for remaining coordinates
    tail_Ds = [
        torch.linspace(1.0, 2.5, max(d - 4, 0), device=device, dtype=dtype),
        torch.linspace(2.5, 0.9, max(d - 4, 0), device=device, dtype=dtype),
        torch.full((max(d - 4, 0),), 1.6, device=device, dtype=dtype),
        torch.linspace(0.7, 2.8, max(d - 4, 0), device=device, dtype=dtype),
    ]

    val = torch.zeros(N, device=device, dtype=dtype)
    sharpness = 200.0

    for amp, c, Axy, A34, Dtail in zip(amps, centers, Axy_list, A34_list, tail_Ds):
        diff = x - c
        r = torch.zeros(N, device=device, dtype=dtype)

        # first rotated block: coordinates (0,1)
        if d >= 2:
            diff_xy = diff[:, 0:2]
            r += torch.einsum("ni,ij,nj->n", diff_xy, Axy, diff_xy)
        elif d == 1:
            r += 3.0 * diff[:, 0] ** 2

        # second rotated block: coordinates (2,3)
        if d >= 4:
            diff_34 = diff[:, 2:4]
            r += torch.einsum("ni,ij,nj->n", diff_34, A34, diff_34)
        elif d == 3:
            r += 2.0 * diff[:, 2] ** 2

        # remaining coordinates diagonal
        if d > 4:
            diff_tail = diff[:, 4:]
            r += (diff_tail * Dtail * diff_tail).sum(dim=1)

        val += amp * torch.exp(-sharpness * r)

    return val

def easy_four_gaussian(x):
    """
    Easier 4-peak Gaussian benchmark on [0,1]^d.

    x: (N, d)
    returns: (N,)
    """
    N, d = x.shape
    device = x.device
    dtype = x.dtype

    centers = torch.stack([
        torch.full((d,), 0.25, device=device, dtype=dtype),
        torch.full((d,), 0.75, device=device, dtype=dtype),
        torch.cat([
            torch.full((d // 2,), 0.25, device=device, dtype=dtype),
            torch.full((d - d // 2,), 0.75, device=device, dtype=dtype),
        ]),
        torch.cat([
            torch.full((d // 2,), 0.75, device=device, dtype=dtype),
            torch.full((d - d // 2,), 0.25, device=device, dtype=dtype),
        ]),
    ], dim=0)

    # Different importance
    amps = torch.tensor([1.0, 0.8, 1.2, 0.6], device=device, dtype=dtype)

    # Different radii: larger sigma = wider/easier peak
    sigmas = torch.tensor([0.12, 0.16, 0.10, 0.18], device=device, dtype=dtype)

    val = torch.zeros(N, device=device, dtype=dtype)

    for amp, c, sigma in zip(amps, centers, sigmas):
        r2 = ((x - c) ** 2).sum(dim=1)
        val += amp * torch.exp(-r2 / sigma**2)

    return val