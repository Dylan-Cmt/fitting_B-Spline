import numpy as np
from math import comb


##
# function: bernstein_basis
#
# description:
#   Computes a Bernstein basis polynomial:
#
#   B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
#
# input:
#   - i : Bernstein basis index
#   - n : Bézier polynomial degree
#   - t : parameter in [0,1]
#
# output:
#   - value of the Bernstein basis B_i^n(t)
#
# errors:
#   - ValueError if n < 0
##
def bernstein_basis(i, n, t):
    if n < 0:
        raise ValueError("Bezier degree must be non-negative")
    if i < 0 or i > n:
        return 0.0
    return comb(n, i) * (t**i) * ((1 - t) ** (n - i))


##
# function: bernstein_basis_vector
#
# description:
#   Generates the complete vector of Bernstein
#   basis functions:
#
#   [B_0^n(t), ..., B_n^n(t)]
#
# input:
#   - n : Bézier polynomial degree
#   - t : scalar or array of parameters
#
# output:
#   - numpy array containing all Bernstein bases
##
def bernstein_basis_vector(n, t):
    t = np.asarray(t, dtype=float)
    basis = np.array([bernstein_basis(i, n, t) for i in range(n + 1)])
    return basis


##
# function: eval_bezier_curve
#
# description:
#   Evaluates a Bézier curve from its control
#   points:
#
#   P(t) = Σ B_i^n(t) * P_i
#
# input:
#   - control_points : array of control points
#   - t : scalar or array of parameters
#
# output:
#   - curve point if t is scalar
#   - array of curve points if t is a vector
#
# errors:
#   - ValueError if control points are empty
##
def eval_bezier_curve(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n < 0:
        raise ValueError("Control points cannot be empty")
    t = np.asarray(t, dtype=float)
    # Scalar case
    if t.ndim == 0:
        basis = bernstein_basis_vector(n, t)
        return basis @ P
    # Vector case
    curve = np.zeros((len(t), P.shape[1]))
    for idx, tt in enumerate(t):
        basis = bernstein_basis_vector(n, tt)
        curve[idx] = basis @ P
    return curve


##
# function: eval_bezier_derivative
#
# description:
#   Computes the first derivative of a Bézier curve.
#
#   The derivative corresponds to the hodograph:
#
#   P'(t) = n * Σ B_i^(n-1)(t) * (P_(i+1) - P_i)
#
# input:
#   - control_points : array of control points
#   - t : scalar or array parameter
#
# output:
#   - derivative vector evaluated at t
##
def eval_bezier_derivative(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n <= 0:
        return np.zeros_like(P[0])
    # Hodograph control points
    D = n * (P[1:] - P[:-1])
    return eval_bezier_curve(D, t)


##
# function: eval_bezier_second_derivative
#
# description:
#   Computes the second derivative of a Bézier curve.
#
# input:
#   - control_points : array of control points
#   - t : scalar or array parameter
#
# output:
#   - second derivative evaluated at t
##
def eval_bezier_second_derivative(control_points, t):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n <= 1:
        return np.zeros_like(P[0])
    # First derivative
    D1 = n * (P[1:] - P[:-1])
    # Second derivative
    D2 = (n - 1) * (D1[1:] - D1[:-1])
    return eval_bezier_curve(D2, t)

##
# function: degree_elevate
#
# description:
#   Elevates the degree of a Bézier curve by one.
#
# input:
#   - control_points : array of control points
#
# output:
#   - array of control points for the elevated curve
##
def degree_elevate(control_points):
    P = np.asarray(control_points, dtype=float)
    n = len(P) - 1
    if n < 0:
        raise ValueError("Control points cannot be empty")
    elif n == 0:
        return np.array([P[0], P[0]])
    # Degree elevation formula
    new_control_points = np.zeros((n + 2, P.shape[1]))
    new_control_points[0] = P[0]
    new_control_points[-1] = P[-1]
    for i in range(1, n + 1):
        coeff = i / (n + 1)
        new_control_points[i] = coeff * P[i - 1] + (1 - coeff) * P[i]
    return new_control_points

##
# function: degree_elevate_multiple
#
# description:
#   Elevates the degree of a Bézier curve by k.
#
# input:
#   - control_points : array of control points
#   - k : number of degree elevations
#
# output:
#   - array of control points for the elevated curve
##
def degree_elevate_multiple(control_points, k):
    P = np.asarray(control_points, dtype=float)
    for _ in range(k):
        P = degree_elevate(P)
    return P


##
# function: unit_tangent
#
# description:
#   Computes the unit tangent vector
#   of the curve.
#
# input:
#   - control_points : array of control points
#   - t : parameter in [0,1]
#
# output:
#   - unit tangent vector
##
def unit_tangent(control_points, t):
    degree = len(control_points) - 1
    if degree < 1:
        raise ValueError("Invalid Bézier degree: tangent computation requires degree >= 1.")
    tangent = eval_bezier_derivative(control_points, t)
    epsilon = 1e-12
    return np.asarray(tangent / (np.linalg.norm(tangent) + epsilon),dtype=float)


##
# function: unit_normal
#
# description:
#   Computes the 2D unit normal vector
#   from the tangent vector.
#
# input:
#   - control_points : array of control points
#   - t : parameter in [0,1]
#
# output:
#   - 2D unit normal vector
##
def unit_normal(control_points, t):
    degree = len(control_points) - 1
    u_T = unit_tangent(control_points, t)
    return np.asarray([u_T[1], -u_T[0]], dtype=float)


##
# function: curvature2D
#
# description:
#   Computes the curvature of a 2D Bézier curve.
#
# input:
#   - control_points : 2D control points
#   - t : parameter in [0,1]
#
# output:
#   - curvature value
##
def curvature2D(control_points, t):
    P_t_prime = eval_bezier_derivative(control_points, t)
    P_t_second = eval_bezier_second_derivative(control_points, t)
    denom = np.linalg.norm(P_t_prime)**3
    # Avoid division by zero
    if denom < 1e-12:
        return 1e12
    num = np.linalg.det(np.vstack((P_t_prime, P_t_second)))
    return abs(num) / denom


##
# function: curvature_radius
#
# description:
#   Computes the radius of curvature.
#
# input:
#   - control_points : array of control points
#   - t : parameter in [0,1]
#
# output:
#   - radius of curvature
##
def curvature_radius(control_points, t):
    K = curvature2D(control_points, t)
    if K < 1e-12:
        return 1e12
    elif K >= 1e12:
        return 0
    return 1 / K