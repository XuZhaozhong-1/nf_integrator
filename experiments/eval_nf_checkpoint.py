import os
import json
import torch

from benchmarks.registry import get_benchmark
from nf.cube_flow import CubeRealNVP
from nf.evaluate import nf_integrate

DIM = 2
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

N_EVENTS = list(range(10000, 100001, 10000))


def evaluate_checkpoint(checkpoint_path, method_name, out_path,n_blocks=12,hidden=128, eps_mix=0.0):
    model = CubeRealNVP(
        dim=DIM,
        n_blocks=n_blocks,
        hidden=hidden,
        eps_mix=eps_mix,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    results = []

    for i, n in enumerate(N_EVENTS):
        torch.manual_seed(i)

        res = nf_integrate(
            model=model,
            benchmark_fn=benchmark_fn,
            n_events=n,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        res["trial"] = i
        res["method"] = method_name
        res["benchmark"] = BENCHMARK
        res["dim"] = DIM
        res["eps_mix"] = eps_mix

        results.append(res)
        print(f"{method_name} | run {i}, N={n} -> {res}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {out_path}")


def main():
    # evaluate_checkpoint(
    #     checkpoint_path=None,
    #     method_name="nf_untrained",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_untrained_sweep.json",
    # )

    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl.pt",
        method_name="nf_A_kl",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_sweep.json",
    )
    
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl_floor001.pt",
        method_name="nf_A_kl_floor001",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor001_sweep.json",
        eps_mix=0.01
    )
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl_floor005.pt",
        method_name="nf_A_kl_floor005",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor005_sweep.json",
        eps_mix=0.05
    )
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl_floor010.pt",
        method_name="nf_A_kl_floor010",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010_sweep.json",
        eps_mix=0.10
    )
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl_floor010-setting2.pt",
        method_name="nf_A_kl_floor010",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010-setting2_sweep.json",
        n_blocks=20,
        hidden=256,
        eps_mix=0.10
    )

    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_B_raw_std.pt",
    #     method_name="nf_B_logw_std",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_B_raw_std_sweep.json",
    # )

    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_C_kl_raw_std.pt",
    #     method_name="nf_C_kl_raw_std",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_C_kl_raw_std_sweep.json",
    # )


if __name__ == "__main__":
    main()