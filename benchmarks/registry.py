from benchmarks.functions import (
    gaussian_family,
    double_gaussian,
    slashed_circle,
    easy_four_gaussian,
)

BENCHMARKS = {
    "gaussian_family": gaussian_family,
    "double_gaussian": double_gaussian,
    "slashed_circle": slashed_circle,
    "easy_four_gaussian": easy_four_gaussian,
}


def get_benchmark(name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name}")
    return BENCHMARKS[name]