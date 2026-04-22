import numpy as np
from math import comb

def bernstein_basis(i, n, t):
    if n < 0:
        raise ValueError("Bezier degree must be non-negative")
    if i < 0 or i > n:
        return 0.0
    return comb(n, i) * (t**i) * ((1 - t) ** (n - i))


def bernstein_basis_vector(n, t):
    t = np.asarray(t, dtype=float)
    basis = np.array([bernstein_basis(i, n, t) for i in range(n + 1)])
    return basis


def eval_bezier_curve(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n < 0:
        raise ValueError("Control points cannot be empty")

    t = np.asarray(t, dtype=float)
    if t.ndim == 0: # equivalent to t.shape[1]
        basis = bernstein_basis_vector(n, t)
        return basis @ P

    curve = np.zeros((len(t), P.shape[1]))
    for idx, tt in enumerate(t):
        basis = bernstein_basis_vector(n, tt)
        curve[idx] = basis @ P
    return curve


def eval_bezier_derivative(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n <= 0:
        return np.zeros_like(P[0])

    D = n * (P[1:] - P[:-1]) # hodograph on [0;1]
    return eval_bezier_curve(D, t)


def eval_bezier_second_derivative(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n <= 1:
        return np.zeros_like(P[0])

    D1 = n * (P[1:] - P[:-1])
    D2 = (n - 1) * (D1[1:] - D1[:-1])
    return eval_bezier_curve(D2, t)
