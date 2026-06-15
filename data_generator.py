#import os
import numpy as np
from src.optimizers.bezier_curves import eval_bezier_curve
from src.optimizers.bezier_optimizer import *
#from src.optimizers.cloud_points import *
import torch
import time

##
# function: cross2d
#
# description:
#   Performs the cross product of two 2D
#   vectors a and b.
#
# input:
#   - a: 2D vector
#   - b: 2D vector
#
# output:
#   - scalar value representing the cross product of a and b
##
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


##
# function: has_self_intersection
#
# description:
#   Checks if a curve has self-intersections.
#
# input:
#   - curve: array of 2D points representing the curve
#   - tol: tolerance for intersection detection
#
# output:
#   - boolean indicating whether the curve has self-intersections
##
def has_self_intersection(curve, tol=1e-4):
    n = len(curve) # number of discrete points in the curve
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            # Check if the segments (curve[i], curve[i+1]) and (curve[j], curve[j+1]) intersect
            p, r = curve[i], curve[i+1] - curve[i]
            q, s = curve[j], curve[j+1] - curve[j]
            denom = cross2d(r, s)
            if abs(denom) < 1e-12:
                continue
            t = cross2d(q - p, s) / denom
            u = cross2d(q - p, r) / denom
            if 0 < t < 1 and 0 < u < 1:
                return True
    return False

##
# function: random_control_points
#
# description:
#   Generates random control points for a Bézier curve
#   and check if the resulting curve has self-intersections.
#
# input:
#   - n: number of control points
#   - max_tries: maximum number of attempts to generate 
#     non-self-intersecting curve
#
# output:
#   - array of control points
##
def random_control_points(n=4, max_tries=100):
    for _ in range(max_tries):
        P = np.random.uniform(0, 1, size=(n, 2))
        t = np.linspace(0, 1, 1000)
        curve = eval_bezier_curve(P, t)
        if not has_self_intersection(curve):
            return P
    raise RuntimeError(f"Can't generate non-self-intersecting control points after {max_tries} tries, please increase number of samplesor increase max_tries.")

##
# function: generate_noised_Bezier_data
#
# description:
#   Generates a dataset  from a noisy Bézier curve 
#   defined by the given control points.
#
# input:
#   - control_points: array of control points
#   - n_points: number of points of the cloud points
#   - noise_std: standard deviation of the noise to add
#
# output:
#   - cloud points 
##
def generate_noised_Bezier_data(control_points, n_points=100, noise_std=0.0):
    t = np.linspace(0, 1, n_points)
    curve = eval_bezier_curve(control_points, t)
    noise = np.random.normal(loc=0.0,scale=noise_std,size=curve.shape)
    return curve + noise

# start generating the dataset
"""
# in case I do not want to erase the old dataset:
if os.path.exists("dataset.pt"):
    dataset_old = torch.load("dataset.pt")
else:
    dataset_old = []
dataset_old.extend(dataset)
"""
dataset = []

dataset_size = 100
counter = 0

for i in range(dataset_size):

    Pc = random_control_points(6)
    X = generate_noised_Bezier_data(Pc,n_points=100,noise_std=0.01)
    #Pc[0] = X[0]
    #Pc[-1] = X[-1]
    T = all_tk(X, Pc)

    max_iter = 100

    Popt, log_iter, log_avg_error, log_max_error = gradient_descent_PD(Pc,T,X, alpha0=0.1,max_iter=max_iter)
    if int(10**log_iter[-1]) == max_iter:
        counter +=1

    dataset.append({
        "X": torch.tensor(X,dtype=torch.float32),
        "initial_control_points": torch.tensor(Pc,dtype=torch.float32),
        "final_control_points": torch.tensor(Popt, dtype=torch.float32)
    })

print(f"{counter} curves reached max iterations")

torch.save(dataset, "dataset.pt")