import contextlib # used to ignore the outputs of the gradient descent function
import numpy as np
import os
from src.optimizers.bezier_curves import eval_bezier_curve
from src.optimizers.bezier_optimizer import *
#from src.optimizers.cloud_points import *
import torch
import time
from tqdm import tqdm # used to display a progress bar for the dataset generation

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
#   The curve is represented as a sequence of points, where consecutive
#   points define line segments. The algorithm tests all pairs of
#   non-adjacent segments and computes their intersection using a
#   parametric formulation:
#
#       p + t r = q + u s
#
#   where:
#       - p, q are segment starting points
#       - r, s are segment direction vectors
#       - t, u are scalar parameters
#
#   If a solution exists with 0 < t < 1 and 0 < u < 1, the two segments
#   intersect strictly inside their endpoints, meaning the curve has a
#   self-intersection.
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
    for i in range(max_tries):
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

##
# function: generate_sample
#
# description:
#   Generates a single sample of the dataset, consisting of a noisy Bézier curve
#   and the optimized control points obtained from gradient descent.
#
# input:
#   - _: placeholder for compatibility with parallel processing
# output:
#   - dictionary containing the noisy curve points and the optimized control points
##
def generate_sample(_):
    Pc = random_control_points(3)
    X = generate_noised_Bezier_data(Pc,n_points=100,noise_std=0.01)
    T = all_tk(X, Pc)

    max_iter = 100
    # the two lines below are to avoid printing the output of the gradient descent function
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f):
            Popt, *_  = gradient_descent_PD(Pc,T,X, alpha0=0.1,max_iter=max_iter)
    return {
    "X": X.astype(np.float32),
    "final_control_points": Popt.astype(np.float32)
}

if __name__ == "__main__":
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

    dataset_size = 10
    
    start_time = time.time()
    
    dataset = list(
    tqdm(
        map(generate_sample, range(dataset_size)),
        total=dataset_size,
        )
    )
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds")

    torch.save(dataset, "./src/IA/datasets/dataset10k3pts.pt")