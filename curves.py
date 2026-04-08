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
        return 1.0 if nodes[i] <= t < nodes[i + 1] else 0.0
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


##############################################################################################
##                                                                                          ##
##                                 foot points calculation                                  ##
##                                                                                          ##
##############################################################################################

# Compute the tangent unit vector of a B-Spline curve at a given parameter t
def tangent_unit_vector(degree, nodes, control_points, t):
    d1 = evalbsplinecurve_derivative_at_t(degree, nodes, control_points, t)
    norm_d1 = np.linalg.norm(d1)
    if norm_d1 == 0:
        return np.zeros_like(d1)
    return d1 / norm_d1

# Compute the normal unit vector of a B-Spline curve at a given parameter t
def normal_unit_vector(degree, nodes, control_points, t):
    tangent = tangent_unit_vector(degree, nodes, control_points, t)
    normal = np.array([-tangent[1], tangent[0]])
    return normal


