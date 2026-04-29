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

def unit_tangent(control_points, t):
    tangent = eval_bezier_derivative(control_points, t)
    epsilon = 1e-12
    return tangent / (np.linalg.norm(tangent) + epsilon)


"""
def unit_normal(control_points, t):
    normal = eval_bezier_second_derivative(control_points, t)
    return normal / np.linalg.norm(normal)
"""
def unit_normal(control_points, t):
    u_T = unit_tangent(control_points, t)
    return [u_T[1], -u_T[0]]



"""
Pc = np.array([
        [0.0, 0.0],
        [-0.5, 1.5],
        [0.5, 1.0],
        [1.5, 1.5],
        [1.0, 0.0]
    ])
t = np.linspace(0,1,100)
c = eval_bezier_curve(Pc,t)
T = unit_tangent(Pc, 0.7 )
N = unit_normal(Pc, 0.7 )
assert np.isclose(np.dot(T, N), 0)
assert np.isclose(np.linalg.det(np.vstack((N,T))),1)
"""
