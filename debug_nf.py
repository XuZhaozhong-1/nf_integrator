import torch

from nf.cube_flow import CubeRealNVP


def constant_fn(x):
    return torch.ones(x.shape[0], device=x.device)


def prod_fn(x):
    return x.prod(dim=1)


def nf_integrate(model, benchmark_fn, n_events: int, device: str = "cpu"):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = model.sample(n_events, device=device)
        f_vals = benchmark_fn(x)
        log_q = model.log_prob(x)
        weights = f_vals * torch.exp(-log_q)

        estimate = weights.mean()
        variance = weights.var(unbiased=True) / n_events

    return estimate.item(), variance.item()


def main():
    device = "cpu"
    torch.manual_seed(0)

    model = CubeRealNVP(dim=2, n_blocks=4, hidden=64).to(device)
    model.eval()

    print("=" * 80)
    print("1. RealNVP roundtrip test")
    print("=" * 80)
    z = torch.randn(5000, model.dim, device=device)
    y, ld_fwd = model.flow.fwd(z)
    z2, ld_inv = model.flow.inv(y)

    print("max |z - inv(fwd(z))| =", (z - z2).abs().max().item())
    print("mean |ld_fwd + ld_inv| =", (ld_fwd + ld_inv).abs().mean().item())
    print("max  |ld_fwd + ld_inv| =", (ld_fwd + ld_inv).abs().max().item())

    print("\n" + "=" * 80)
    print("2. Cube transform roundtrip test")
    print("=" * 80)
    x = torch.sigmoid(y)
    y2 = torch.log(x) - torch.log1p(-x)
    print("max |y - logit(sigmoid(y))| =", (y - y2).abs().max().item())

    print("\n" + "=" * 80)
    print("3. Density normalization test on unit cube")
    print("=" * 80)
    xu = torch.rand(200000, model.dim, device=device)
    log_q_u = model.log_prob(xu)
    q_u = torch.exp(log_q_u)
    print("Uniform MC estimate of ∫ q(x) dx over cube =", q_u.mean().item())
    print("This should be close to 1.0")

    print("\n" + "=" * 80)
    print("4. Constant-function integral test")
    print("=" * 80)
    est_const, var_const = nf_integrate(model, constant_fn, 50000, device=device)
    print("NF estimate for ∫1 dx =", est_const)
    print("Variance =", var_const)
    print("This should be close to 1.0")

    print("\n" + "=" * 80)
    print("5. Known integral test: prod(x)")
    print("=" * 80)
    est_prod, var_prod = nf_integrate(model, prod_fn, 50000, device=device)
    true_prod = (0.5) ** model.dim
    print("NF estimate for ∫ prod(x) dx =", est_prod)
    print("True value =", true_prod)
    print("Variance =", var_prod)

    print("\n" + "=" * 80)
    print("6. Boundary saturation test")
    print("=" * 80)
    xs = model.sample(100000, device=device)
    print("sample min =", xs.min().item())
    print("sample max =", xs.max().item())
    print("frac(x < 1e-6) =", (xs < 1e-6).float().mean().item())
    print("frac(x > 1-1e-6) =", (xs > 1 - 1e-6).float().mean().item())

    print("\n" + "=" * 80)
    print("7. Sample/log_prob sanity")
    print("=" * 80)
    log_q_s = model.log_prob(xs)
    print("log_q on sampled x: mean =", log_q_s.mean().item())
    print("log_q on sampled x: min  =", log_q_s.min().item())
    print("log_q on sampled x: max  =", log_q_s.max().item())
    print("Any NaN in log_q?", torch.isnan(log_q_s).any().item())
    print("Any inf in log_q?", torch.isinf(log_q_s).any().item())


if __name__ == "__main__":
    main()
