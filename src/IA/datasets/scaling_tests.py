import contextlib # used to ignore the outputs of the gradient descent function
from concurrent.futures import ProcessPoolExecutor # used to parallelize the dataset generation
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from src.optimizers.bezier_curves import eval_bezier_curve, degree_elevate_multiple
from src.optimizers.bezier_optimizer import *
#from src.optimizers.cloud_points import *
import torch
import time
from tqdm import tqdm # used to display a progress bar for the dataset generation

def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

def has_self_intersection(curve, tol=1e-4):
    n = len(curve) # number of discrete points in the curve
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            # Check if the segments [p, p+r] and [q, q+s] intersect
            p, r = curve[i], curve[i+1] - curve[i]
            q, s = curve[j], curve[j+1] - curve[j]
            denom = cross2d(r, s)
            if abs(denom) < 1e-12:
                continue
            # The segments are configured by p+tr,0≤t≤1 and q+us,0≤u≤1
            # We want to solve p+tr=q+us to find intersect (2eq, 2var because 2D so we find)
            t = cross2d(q - p, s) / denom
            u = cross2d(q - p, r) / denom
            # Then we check if t and u are between 0 and 1 as parameters (else, interset doesn't occur)
            if 0 < t < 1 and 0 < u < 1:
                return True
    return False

def random_control_points(n=4, max_tries=100):
    for i in range(max_tries):
        P = np.random.uniform(0, 1, size=(n, 2))
        t = np.linspace(0, 1, 1000)
        curve = eval_bezier_curve(P, t)
        if not has_self_intersection(curve):
            return P
    raise RuntimeError(f"Can't generate non-self-intersecting control points after {max_tries} tries, please increase number of samplesor increase max_tries.")

def generate_noised_Bezier_data(control_points, n_points=100, noise_std=0.0):
    t = np.linspace(0, 1, n_points)
    curve = eval_bezier_curve(control_points, t)
    noise = np.random.normal(loc=0.0,scale=noise_std,size=curve.shape)
    return curve + noise

def generate_sample(_):
    degree = random.randint(2, 5)
    Pc = random_control_points(degree + 1)
    X = generate_noised_Bezier_data(Pc,n_points=100,noise_std=0.01)
    T = all_tk(X, Pc)

    max_iter = 100

    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f):
            Popt, *_ = gradient_descent_PD(Pc,T,X,alpha0=0.1, max_iter=max_iter)
    if degree < 5:
        Popt = degree_elevate_multiple(Popt, 5 - degree)
    return {
        "X": X.astype(np.float32),
        "final_control_points": Popt.astype(np.float32)
    }

def run_benchmark(dataset_size, workers):
    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        dataset = list(
            tqdm(
                executor.map(generate_sample, range(dataset_size)),
                total=dataset_size,
                leave=False
            )
        )

    dataset = [
        {
            "X": torch.from_numpy(d["X"]),
            "final_control_points": torch.from_numpy(d["final_control_points"])
        }
        for d in dataset
    ]

    end_time = time.perf_counter()
    return end_time - start_time


if __name__ == "__main__":


    workers_list = [1, 2, 4, 6, 8]
    n_runs = 3

    print("\n========================")
    print("STRONG SCALING BENCHMARK")
    print("========================")

    dataset_size = 1000

    strong_mean = {}
    strong_std = {}

    for workers in workers_list:
        times = []

        print(f"\n=== TEST {workers} WORKERS ===")

        for run in range(n_runs):
            print(f"run {run+1}/{n_runs}")
            t = run_benchmark(dataset_size, workers)
            times.append(t)
            print(f"  -> {t:.2f}s")

        strong_mean[workers] = np.mean(times)
        strong_std[workers] = np.std(times)

        print(f"==> mean: {strong_mean[workers]:.2f}s ± {strong_std[workers]:.2f}s")

    # ---- PLOT ----
    x = list(strong_mean.keys())
    y = [strong_mean[w] for w in x]
    err = [strong_std[w] for w in x]

    plt.figure()
    plt.errorbar(x, y, yerr=err, marker="o", capsize=5)
    plt.xlabel("Number of workers")
    plt.ylabel("Time (s)")
    plt.title("Strong Scaling")
    plt.grid(True)

    plt.savefig("./src/IA/datasets/strong_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n========================")
    print("WEAK SCALING BENCHMARK")
    print("========================")

    samples_per_worker = 100

    weak_mean = {}
    weak_std = {}

    for workers in workers_list:

        dataset_size = workers * samples_per_worker

        times = []

        print(
            f"\n=== TEST {workers} WORKERS "
            f"({dataset_size} samples) ==="
        )

        for run in range(n_runs):
            print(f"run {run+1}/{n_runs}")
            t = run_benchmark(dataset_size, workers)
            times.append(t)
            print(f"  -> {t:.2f}s")

        weak_mean[workers] = np.mean(times)
        weak_std[workers] = np.std(times)

        print(
            f"==> mean: {weak_mean[workers]:.2f}s "
            f"± {weak_std[workers]:.2f}s"
        )

    # ---- PLOT ----
    x = list(weak_mean.keys())
    y = [weak_mean[w] for w in x]
    err = [weak_std[w] for w in x]

    plt.figure()
    plt.errorbar(x, y, yerr=err, marker="o", capsize=5)
    plt.xlabel("Number of workers")
    plt.ylabel("Time (s)")
    plt.title("Weak Scaling")
    plt.grid(True)

    plt.savefig("./src/IA/datasets/weak_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()