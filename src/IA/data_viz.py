import matplotlib.pyplot as plt
import numpy as np
from src.optimizers.bezier_curves import eval_bezier_curve
from src.optimizers.cloud_points import generate_sinus_cloud_points
import torch

dataset = torch.load("src/IA/datasets/dataset10k3pts.pt")


for i in range(len(dataset)):
    X = dataset[i]["X"].numpy()
    Popt = dataset[i]["Popt"].numpy()

    bz_opt = eval_bezier_curve(Popt, np.linspace(0,1,100))
    # plot initial and optimized curves with control points and cloud points
    plt.figure(figsize=(12, 9))
    plt.scatter(X[:, 0], X[:, 1], color='green',  alpha=0.5, s=30, label='Cloud Points')

    #plt.plot(Pc[:, 0], Pc[:, 1], color='red', linestyle='--',linewidth=1, label='Initial Control Polygon')

    plt.scatter(Popt[:, 0], Popt[:, 1], color='blue', marker='x', s=50, label='Optimized Control Points')
    #plt.plot(Popt[:, 0], Popt[:, 1], color='blue', linestyle='--', linewidth=1, label='Optimized Control Polygon')
    plt.plot(bz_opt[:, 0], bz_opt[:, 1], color='blue', linewidth=1, label='Optimized Curve')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()