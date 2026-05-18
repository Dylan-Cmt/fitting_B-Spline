import numpy as np

##
# function: bspline_basis
#
# description:
#   Computes the B-spline basis function
#   N_i,p(t) using the Cox–de Boor
#   recursive formula.
#
# input:
#   - i : basis function index
#   - degree : spline degree
#   - t : parameter value
#   - knots : knot vector
#
# output:
#   - value of the B-spline basis function
#     N_i,p(t)
##
def bspline_basis(i, degree, t, knots):
    n = len(knots) - 1
    if degree == 0:
        if i >= n:
            return 0.0
        return 1.0 if knots[i] <= t < knots[i + 1] or (t == knots[-1] and knots[i + 1] == knots[-1]) else 0.0
    first_part = 0.0
    second_part = 0.0
    if (i + degree) < n:
        denom1 = knots[i + degree] - knots[i]
        if denom1 != 0:
            first_part = (t - knots[i]) / denom1 * bspline_basis(i, degree - 1, t, knots)
    if (i + degree + 1) < n:
        denom2 = knots[i + degree + 1] - knots[i + 1]
        if denom2 != 0:
            second_part = (
                (knots[i + degree + 1] - t)
                / denom2
                * bspline_basis(i + 1, degree - 1, t, knots)
            )
    return first_part + second_part
 

 ##
# function: dt_bspline_basis
#
# description:
#   Computes the first derivative of a B-spline
#   basis function using the standard formula
#   from the NURBS Book (p. 59).
#
# input:
#   - i : basis function index
#   - degree : spline degree (p)
#   - t : parameter value
#   - knots : knot vector
#
# output:
#   - value of the first derivative N'_{i,p}(t)
##
def dt_bspline_basis(i, degree, t, knots): # cf p59 du NURBS BOOK
    first_part, second_part = 0.0, 0.0
    denom1 = knots[i + degree] - knots[i]
    if denom1 != 0:
        first_part  = degree / denom1 * bspline_basis(i, degree - 1, t, knots)
    denom2 = knots[i + degree + 1] - knots[i + 1]
    if denom2 != 0:
        second_part = degree / denom2 * bspline_basis(i + 1, degree - 1, t, knots)
    return first_part - second_part


# function: dtt_bspline_basis
#
# description:
#   Computes the second derivative of a B-spline
#   basis function using the standard formula
#   from the NURBS Book (p. 62).
#
# input:
#   - i : basis function index
#   - degree : spline degree (p)
#   - t : parameter value
#   - knots : knot vector
#
# output:
#   - value of the second derivative
#     N''_{i,p}(t)
##
def dtt_bspline_basis(i, degree, t, knots): # cf p62 du NURBS BOOK
    first_part, second_part, third_part = 0.0, 0.0, 0.0

    # N_{i, p-2}
    denom1 = (knots[i + degree] - knots[i]) * (knots[i + degree - 1] - knots[i])
    if denom1 != 0:
        first_part = degree * (degree - 1) / denom1 * bspline_basis(i, degree - 2, t, knots)

    # N_{i+1, p-2}
    denom2a = (knots[i + degree] - knots[i]) * (knots[i + degree] - knots[i + 1])
    denom2b = (knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree] - knots[i + 1])

    coeff2 = 0.0
    if denom2a != 0:
        coeff2 += 1.0 / denom2a
    if denom2b != 0:
        coeff2 += 1.0 / denom2b

    second_part = degree * (degree - 1) * coeff2 * bspline_basis(i + 1, degree - 2, t, knots)

    # N_{i+2, p-2}
    denom3 = (knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree + 1] - knots[i + 2])
    if denom3 != 0:
        third_part = degree * (degree - 1) / denom3 * bspline_basis(i + 2, degree - 2, t, knots)

    return first_part - second_part + third_part

 

##
# function: bspline_curve
#
# description:
#   Evaluates a B-spline curve at parameter t
#   using the basis function expansion:
#
#   C(t) = Σ N_{i,p}(t) * P_i
#
# input:
#   - P : control points (array of dimension n x d)
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree p
#
# output:
#   - point on the B-spline curve at t
##
def bspline_curve(P, t, knots, degree):
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = bspline_basis(i, degree, t, knots)
        curve += N * P[i]
    return curve



##
# function: dt_bspline_curve
#
# description:
#   Evaluates the first derivative of a B-spline
#   curve at parameter t:
#
#   C'(t) = Σ N'_{i,p}(t) * P_i
#
# input:
#   - P : control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - tangent vector of the curve at t
##
def dt_bspline_curve(P, t, knots, degree):
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = dt_bspline_basis(i, degree, t, knots)
        curve += N * P[i]
    return curve


##
# function: dtt_bspline_curve
#
# description:
#   Evaluates the second derivative of a B-spline
#   curve at parameter t:
#
#   C''(t) = Σ N''_{i,p}(t) * P_i
#
# input:
#   - P : control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - second derivative vector of the curve
##
def dtt_bspline_curve(P, t, knots, degree):
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = dtt_bspline_basis(i, degree, t, knots)
        curve += N * P[i]
    return curve


##
# function: unit_tangent
#
# description:
#   Computes the unit tangent vector of a B-spline
#   curve at parameter t.
#
# input:
#   - control_points : spline control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - unit tangent vector
##
def unit_tangent(control_points, t, knots, degree):
    tangent = dt_bspline_curve(control_points, t, knots, degree)
    epsilon = 1e-12
    return np.asarray(tangent / (np.linalg.norm(tangent) + epsilon), dtype=float)



##
# function: unit_normal
#
# description:
#   Computes the 2D unit normal vector from the
#   unit tangent vector.
#
# input:
#   - control_points : spline control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - unit normal vector in 2D
##
def unit_normal(control_points, t, knots, degree):
    u_T = unit_tangent(control_points, t, knots, degree)
    return np.asarray([u_T[1], -u_T[0]], dtype=float)


##
# function: curvature2D
#
# description:
#   Computes the curvature of a 2D B-spline curve.
#
# input:
#   - control_points : spline control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - curvature value
##
def curvature2D(control_points , t, knots, degree):
    P_t_prime = dt_bspline_curve(control_points, t, knots, degree)
    P_t_second = dtt_bspline_curve(control_points, t, knots, degree)
    denom = np.linalg.norm(P_t_prime)**3
    if (denom < 1e-12):
        return 1e12
    num = np.linalg.det(np.vstack((P_t_prime,P_t_second)))
    return abs(num) / denom


##
# function: curvature_radius
#
# description:
#   Computes the radius of curvature.
#
# input:
#   - control_points : spline control points
#   - t : parameter value
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - curvature radius
##
def curvature_radius(control_points, t, knots, degree):
    K = curvature2D(control_points, t, knots, degree)
    if (K < 1e-12):
        return 1e12
    elif(K >= 1e12):
        return 0
    return 1/K