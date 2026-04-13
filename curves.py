import numpy as np

##############################################################################################
##                                                                                          ##
##                                 B-Spline and its derivatives                             ##
##                                                                                          ##
##############################################################################################
def evalbspline_basis(i, degree, nodes, t):
    n = len(nodes) - 1
    if degree < 0:
        raise ValueError(f"Degree cannot be negative. Received: {degree}")
    if i < 0 or i >= n:
        raise ValueError(
            f"Index i ({i}) is out of bounds for knot VectorN of length {len(nodes)}"
        )
    if degree == 0:
        if i >= n:
            return 0.0
        return 1.0 if nodes[i] <= t < nodes[i + 1] or (t == nodes[-1] and nodes[i + 1] == nodes[-1]) else 0.0
    first_part = 0.0
    second_part = 0.0
    if (i + degree) < n:
        denom1 = nodes[i + degree] - nodes[i]
        if denom1 != 0:
            first_part = (t - nodes[i]) / denom1 * evalbspline_basis(i, degree - 1, nodes, t)
    if (i + degree + 1) < n:
        denom2 = nodes[i + degree + 1] - nodes[i + 1]
        if denom2 != 0:
            second_part = (
                (nodes[i + degree + 1] - t)
                / denom2
                * evalbspline_basis(i + 1, degree - 1, nodes, t)
            )
    return first_part + second_part

def evalbsplinecurve_at_t(degree, nodes, control_points, t):
    t_min = nodes[degree]
    t_max = nodes[-degree - 1]
    if t < t_min or t > t_max:
        raise ValueError(f"Parameter t ({t}) is out of bounds for knot vector with degree {degree}")
    curve_at_t = np.zeros(control_points.shape[1])
    for i in range(len(control_points)):
        N = evalbspline_basis(i, degree, nodes, t)
        curve_at_t += N * control_points[i]
    return curve_at_t

def evalbsplinecurve(degree, nodes, control_points, sample=300):
    t_min = nodes[degree]
    t_max = nodes[-degree - 1]
    t_vals = np.linspace(t_min, t_max, sample)
    curve = np.zeros((sample, control_points.shape[1]))

    for idx, t in enumerate(t_vals):
        curve_at_t = evalbsplinecurve_at_t(degree, nodes, control_points, t)
        curve[idx] = curve_at_t 
    return curve


def hodograph(degree, nodes, control_points):
    n_ctrl = len(control_points)
    dim = control_points.shape[1]

    hodograph_points = np.zeros((n_ctrl - 1, dim))

    for i in range(n_ctrl - 1):
        denom = nodes[i + degree + 1] - nodes[i + 1]
        if denom == 0:
            hodograph_points[i] = 0
        else:
            hodograph_points[i] = degree * (control_points[i + 1] - control_points[i]) / denom

    return hodograph_points

def evalbsplinecurve_derivative(degree, nodes, control_points, sample=300):
    if degree < 0:
        raise ValueError(f"Degree cannot be negative. Received: {degree}")
    if degree == 0:
        return np.zeros(control_points.shape[1])
    else:
        new_control_points = hodograph(degree, nodes, control_points)
        new_degree = degree - 1
        new_nodes = nodes[1:-1]

    return evalbsplinecurve(new_degree, new_nodes, new_control_points, sample)

def evalbsplinecurve_derivative_at_t(degree, nodes, control_points, t):
    if degree == 0:
        return np.zeros(control_points.shape[1])
    return evalbsplinecurve_at_t(degree - 1, nodes[1:-1], hodograph(degree, nodes, control_points), t)

def evalbsplinecurve_second_derivative(degree, nodes, control_points, sample=300):
    if degree < 2:
        return np.zeros(control_points.shape[1])
    first_derivative_control_points = hodograph(degree, nodes, control_points)
    second_derivative_control_points = hodograph(degree - 1, nodes[1:-1], first_derivative_control_points)
    return evalbsplinecurve(degree - 2, nodes[2:-2], second_derivative_control_points, sample)

def evalbsplinecurve_second_derivative_at_t(degree, nodes, control_points, t):
    if degree < 2:
        return np.zeros(control_points.shape[1])
    first_derivative_control_points = hodograph(degree, nodes, control_points)
    second_derivative_control_points = hodograph(degree - 1, nodes[1:-1], first_derivative_control_points)
    return evalbsplinecurve_at_t(degree - 2, nodes[2:-2], second_derivative_control_points, t)

##############################################################################################
##                                                                                          ##
##                                 foot points approximation                                ##
##                                                                                          ##
##############################################################################################

"""
L'approximation plus fine de tk avec cette fonction se fera plus tard
"""
def foot_points_approx(bspline, bspline_derivative, Xk):
    # Preprocessing

    # Query

    tk = 1
    return tk

# function to minimize for finding tk
def F(t,Xk,degree, nodes, control_points):
    P = evalbsplinecurve_at_t(degree, nodes, control_points, t)
    return np.linalg.norm(P - Xk) ** 2

# function for which we want to find the 0
def F_prime(t, Xk, degree, nodes, control_points):
    P = evalbsplinecurve_at_t(degree, nodes, control_points, t)
    Pprime = evalbsplinecurve_derivative_at_t(degree, nodes, control_points, t)
    return 2 * np.dot(P - Xk, Pprime)

def F_primprim(t, Xk, degree, nodes, control_points):
    P = evalbsplinecurve_at_t(degree, nodes, control_points, t)
    Pprime = evalbsplinecurve_derivative_at_t(degree, nodes, control_points, t)
    Pprimprim = evalbsplinecurve_second_derivative_at_t(degree, nodes, control_points, t)
    return 2 * np.dot(Pprime, Pprime) + 2 * np.dot(P - Xk, Pprimprim)

def newton(Xk, degree, nodes, control_points, initial_guess, tol=1e-6, max_iter=100):

    tk = initial_guess
    #t_min = nodes[degree]
    #t_max = nodes[-degree - 1]
    #tk = np.clip(initial_guess, t_min, t_max)  # we can clip tk to keep it in a valid range
    for _ in range(max_iter):
        Fp = F_prime(tk, Xk, degree, nodes, control_points)
        Fpp = F_primprim(tk, Xk, degree, nodes, control_points)
        if Fpp == 0:
            break
        step = Fp / Fpp
        # step = np.clip(step, -0.1, 0.1) # we can also limit the step
        tk_new = tk - step
        # tk_new = np.clip(tk_new, t_min, t_max) 
        if (abs(tk_new - tk) < tol) or (abs(Fp) < tol):
            return tk_new
        tk = tk_new
    return tk


##############################################################################################
##                                                                                          ##
##                                 other tools                                              ##
##                                                                                          ##
##############################################################################################

def tangent_unit_vector(degree, nodes, control_points, t):
    d1 = evalbsplinecurve_derivative_at_t(degree, nodes, control_points, t)
    norm_d1 = np.linalg.norm(d1)
    if norm_d1 == 0:
        return np.zeros_like(d1)
    return d1 / norm_d1

def normal_unit_vector(tangent_unit_vector):
    normal = np.array([-tangent_unit_vector[1], tangent_unit_vector[0]])
    return normal

def d(Xk, Ptk):
    return np.linalg.norm(Xk - Ptk)


def curvature_at_tk(Pprime, Pprimprim):
    denom = np.linalg.norm(Pprime) ** 3
    if denom == 0:
        return 0
    return np.linalg.norm(np.cross(Pprime, Pprimprim)) / denom

