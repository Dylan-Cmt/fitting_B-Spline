import matplotlib.pyplot as plt
import numpy as np
from src.optimizers.bezier_curves import eval_bezier_curve
from src.optimizers.cloud_points import generate_sinus_cloud_points
import torch

m = 100
noise = 0.01
X_data = generate_sinus_cloud_points(m, noise)

dataset = torch.load("dataset.pt")

"""
err_min = min([d["log_max_error"] for d in dataset])
err_max = max([d["log_max_error"] for d in dataset])
def score(err, err_min=err_min, err_max=err_max):
    return (10**err_max - 10**err) / 10**err_max
"""

"""
for i in range(len(dataset)):
    Pc = dataset[i]["initial_control_points"].numpy()
    Popt = dataset[i]["final_control_points"].numpy()


    bz_init = eval_bezier_curve(Pc, np.linspace(0,1,100))
    bz = eval_bezier_curve(Popt, np.linspace(0,1,100))
    # plot initial and optimized curves with control points and cloud points
    plt.figure(figsize=(12, 9))
    plt.scatter(X_data[:, 0], X_data[:, 1], color='green',  alpha=0.5, s=30, label='Cloud Points')
    plt.scatter(Pc[:, 0], Pc[:, 1], color='red', marker='x', s=50, label='Initial Control Points')
    plt.plot(Pc[:, 0], Pc[:, 1], color='red', linestyle='--',linewidth=1, label='Initial Control Polygon')
    plt.plot(bz_init[:, 0], bz_init[:, 1], color='red', linewidth=1, label='Initial Curve')
    plt.scatter(Popt[:, 0], Popt[:, 1], color='blue', marker='x', s=50, label='Optimized Control Points')
    plt.plot(Popt[:, 0], Popt[:, 1], color='blue', linestyle='--',linewidth=1, label='Optimized Control Polygon')
    plt.plot(bz[:, 0], bz[:, 1], color='blue', linewidth=1, label='Optimized Curve')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.title(f'Total iterations: {dataset[i]["total_iterations"]} \n Total time: {dataset[i]["total_time"]:.4f} \n Avg log_error: {dataset[i]["log_avg_error"]:.4f} \n Max log_error: {dataset[i]["log_max_error"]:.4f} \n Score final à calculer')
    plt.show()"""
print([d["log_max_error"] for d in dataset])