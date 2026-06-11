#import os
from src.optimizers.bezier_optimizer import *
from src.optimizers.cloud_points import *
import time
import torch

def random_control_points_sorted(n=4, noise=1.0):
    P = np.zeros((n, 2))
    P[0] = [0.0, 0.0]
    P[-1] = [1.0, 0.0]
    P[1:-1, 0] = np.random.uniform(-1, 2, n - 2)
    P[1:-1, 1] = np.random.uniform(-1, 1, n - 2)
    return P

m = 100
noise = 0.04
X = generate_sinus_cloud_points(m, noise) # 50 points without noise

"""
# in case I do not want to erase the old dataset:
if os.path.exists("dataset.pt"):
    dataset_old = torch.load("dataset.pt")
else:
    dataset_old = []
dataset_old.extend(dataset)
"""
dataset = []
dataset_size = 10
for _ in range(dataset_size):

    Pc = random_control_points_sorted(4)

    T = all_tk(X, Pc)
    start_time = time.time()
    optimized_control_points, tot_iter, logavg_error, logmax_error = gradient_descent_PD(Pc, T, X, max_iter=75)
    stopping_time = time.time()
    tot_time = stopping_time - start_time
    dataset.append({
        "initial_control_points": torch.tensor(Pc, dtype=torch.float32),
        "final_control_points": torch.tensor(
            optimized_control_points,
            dtype=torch.float32
        ),
        "log_avg_error": float(logavg_error[-1]), 
        "log_max_error": float(logmax_error[-1]),
        "function_type": "sinus",
        "total_iterations": int(10**float(tot_iter[-1])),
        "total_time": float(tot_time)
    })

torch.save(dataset, "dataset.pt")