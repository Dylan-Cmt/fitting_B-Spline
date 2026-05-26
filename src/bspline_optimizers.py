import numpy as np
from . import bspline_curves as cv


#################################################
#                                               #
#           Basis of all methods                #
#                                               #
#################################################


##
# function: err_PD
#
# description:
#   Computes the squared Euclidean distance
#   between a BSpline curve point P(t)
#   and a data point Xk.
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : Bspline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - squared point distance error
##
def err_PD(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    return np.linalg.norm(P_t - Xk) ** 2


##
# function: err_PD_prime
#
# description:
#   Computes the first derivative of the
#   point distance error with respect to t.
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - first derivative of the error
##
def err_PD_prime(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    P_t_prime = cv.dt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(P_t - Xk, P_t_prime)


##
# function: err_PD_primprim
#
# description:
#   Computes the second derivative of the
#   point distance error with respect to t.
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - second derivative of the error
##
def err_PD_primprim(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    P_t_prime = cv.dt_bspline_curve(control_points, t, knots, degree)
    P_t_second = cv.dtt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(P_t_prime, P_t_prime) + 2 * np.dot(P_t - Xk, P_t_second)


##
# function: newton_tk
#
# description:
#   Uses Newton's method to compute the optimal
#   parameter tk associated with point Xk.
#
# input:
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#   - initial_guess : starting value for Newton iteration
#   - tol : convergence tolerance
#   - max_iter : maximum number of iterations
#
# output:
#   - optimized parameter tk
##
def newton_tk(Xk, control_points, knots, degree, initial_guess,tol=1e-6, max_iter=100):
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    t = np.clip(initial_guess, t_min + 1e-8, t_max - 1e-8)
    for _ in range(max_iter):
        Fp  = err_PD_prime(t, Xk, control_points, knots, degree)
        Fpp = err_PD_primprim(t, Xk, control_points, knots, degree)
        if abs(Fpp) < 1e-14:
            break
        t_new = np.clip(t - Fp / Fpp, t_min + 1e-8, t_max - 1e-8)
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
    return t


##
# function: all_tk
#
# description:
#   Computes all parameter values tk associated
#   with a set of data points X.
#
#   If no initial guesses are provided, a coarse
#   sampling of the curve is used.
#
# input:
#   - X : list of data points
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#   - initial_guesses : optional initial parameter values
#   - n_samples : number of samples for initialization
#
# output:
#   - list of optimized foot points tk for every Xk
##
def all_tk(X, control_points, knots, degree, initial_guesses=None, n_samples=50):
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    # First iteration
    # initial guesses are estimated automatically
    if initial_guesses is None: 
        t_samples = np.linspace(t_min, t_max, n_samples, endpoint=False)
        initial_guesses = []
        for Xk in X:
            dists = [err_PD(t, Xk, control_points, knots, degree) for t in t_samples]
            initial_guesses.append(t_samples[np.argmin(dists)])
    # Next iterations
    # previous tk values are reused
    return [newton_tk(Xk, control_points, knots, degree, t0) for Xk, t0 in zip(X, initial_guesses)]


##
# function: newton_alpha
#
# description:
#   Uses Newton's method to compute the foot point of Xk.
#   A faster two-phase approach (preprocessing and query)
#   could be employed for improved convergence, but
#   Newton's method is preferred here.
#
# input:
#   - d_phi_func : first derivative of line-search function φ
#   - dd_phi_func : second derivative of φ
#   - P : control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#   - alpha0 : initial step size
#   - tol : convergence tolerance
#   - max_iter : maximum number of iterations
#
# output:
#   - optimized alpha value
##
def newton_alpha(d_phi_func, dd_phi_func, P, T, knots, degree, X, alpha0, tol=1e-6, max_iter=100):
    alpha = alpha0
    for _ in range(max_iter):
        grad = d_phi_func(alpha, P, T, knots, degree, X)
        if abs(grad) < tol:
            break
        denom = dd_phi_func(alpha, P, T, knots, degree, X)
        if denom == 0:
            break
        alpha = alpha - grad / denom
    return alpha


##
# function: avg_error
#
# description:
#   Computes the root mean square fitting error.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - average fitting error
##
def avg_error(P, T, knots, degree, X):
    return np.sqrt(2 * f_PD(P, T, knots, degree, X) / len(X))


#################################################
#                                               #
#                      PDM                      #
#                                               #
#################################################


##
# function: f_PD
#
# description:
#   Computes the Point Distance Minimization (PDM)
#   objective function.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - PDM value
##
def f_PD(P, T, knots, degree, X):
    return 0.5 * sum(err_PD(T[k], X[k], P, knots, degree) for k in range(len(T)))


##
# function: dP_f_PD
#
# description:
#   Computes the gradient of the PDM objective
#   function with respect to the control points.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_PD(P, T, knots, degree, X):
    m = P.shape[0]
    n = len(T)
    grad_P = np.zeros_like(P)

    for k in range(n):
        d = cv.bspline_curve(P, T[k], knots, degree) - X[k]
        for i in range(m):
            B_ik = cv.bspline_basis(i, degree, T[k], knots)
            grad_P[i] += d * B_ik
    return grad_P


"""
##
# function: phi_PD
#
# description:
#   Evaluates the line-search function phi
#   used in the Tangent Distance Minimization (PDM).
#   This function is not explicitly used, but
#   its derivatives are required for the
#   Newton line-search procedure.
##
def phi_PD(alpha, P, T, knots, degree, X):
    D = -dP_f_PD(P, T, knots, degree, X)
    return f_PD(P + alpha * D, T, knots, degree, X)
"""


##
# function: d_phi_PD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for PDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_PD(alpha, P, T, knots, degree, X):
    D    = -dP_f_PD(P, T, knots, degree, X)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        r_k = cv.bspline_curve(P + alpha * D, T[k], knots, degree) - X[k]
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        dphi += np.dot(r_k, d_k)
    return dphi


##
# function: dd_phi_PD
#
# description:
#   Computes the second derivative of the
#   line-search function phi for PDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - second derivative of phi
##
def dd_phi_PD(alpha, P, T, knots, degree, X):
    D     = -dP_f_PD(P, T, knots, degree, X)
    ddphi = 0.0
    m = len(T)
    for k in range(m):
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        ddphi += np.dot(d_k, d_k)
    return ddphi


##
# function: gradient_descent_PD
#
# description:
#   Performs gradient descent optimization for
#   the Point Distance Minimization (PDM).
#
#   At each iteration:
#   - the gradient is computed
#   - an optimal step size alpha is estimated
#     using Newton's method
#   - the control points are updated
#   - the footpoint parameters tk are recomputed
#
# input:
#   - P0 : initial control points
#   - T0 : initial parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum number of iterations
#   - tol : convergence tolerance
#   - constraint : whether endpoint constraints
#                  are enforced
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_PD(P0, T0, knots, degree, X, alpha0=0.1, max_iter=100, tol=1e-6, constraint=True):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, knots, degree, X)))
        grad_P = dP_f_PD(P, T, knots, degree, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_PD, dd_phi_PD, P, T, knots, degree, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        if constraint:
            # constraints can be modified in need
            #P[0] = P0[0]
            #P[-1] = P0[-1]
            P[-1] = P[0]
        T = all_tk(X, P, knots, degree, initial_guesses=T)
    return P, log_iter, log_avg_error


#################################################
#                                               #
#                      TDM                      #
#                                               #
#################################################


##
# function: err_TD
#
# description:
#   Computes the Tangent Distance (TD) error
#   between a curve point P(t) and a data
#   point Xk.
#
#   The error corresponds to the squared
#   projection of the difference vector
#   onto the unit normal direction.
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - tangent distance error
##
def err_TD(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    n = cv.unit_normal(control_points, t, knots, degree)
    return np.dot(P_t - Xk, n) ** 2


##
# function: f_TD
#
# description:
#   Computes the global Tangent Distance Minimization
#   (TDM) objective function.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - TDM value
##
def f_TD(P, T, knots, degree, X):
    return 0.5 * sum(err_TD(T[k], X[k], P, knots, degree) for k in range(len(T)))


##
# function: dP_f_TD
#
# description:
#   Computes the gradient of the TDM objective
#   function with respect to the control points.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_TD(P, T, knots, degree, X):
    m = P.shape[0]
    n = len(T)
    grad_P = np.zeros_like(P)

    for k in range(n):
        d_k = cv.bspline_curve(P, T[k], knots, degree) - X[k]
        n_k = cv.unit_normal(P, T[k], knots, degree)
        d_proj = np.dot(n_k, d_k)
        for i in range(m):
            B_ik = cv.bspline_basis(i, degree, T[k], knots)
            grad_P[i] += d_proj * B_ik * n_k
    return grad_P


"""
##
# function: phi_TD
#
# description:
#   Evaluates the line-search function phi
#   used in the Tangent Distance Minimization (TDM).
#   This function is not explicitly used, but
#   its derivatives are required for the
#   Newton line-search procedure.
##
def phi_TD(alpha, P, T, knots, degree, X):
    D = -dP_f_TD(P, T, knots, degree, X)
    return f_TD(P + alpha * D, T, knots, degree, X)
"""


##
# function: d_phi_TD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for TDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_TD(alpha, P, T, knots, degree, X):
    D    = -dP_f_TD(P, T, knots, degree, X)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        n_k = cv.unit_normal(P, T[k], knots, degree)
        r_k = cv.bspline_curve(P + alpha * D, T[k], knots, degree) - X[k]
        d_k = np.dot(n_k, cv.bspline_curve(D, T[k], knots, degree))
        dphi += np.dot(n_k, r_k) * d_k
    return dphi


##
# function: dd_phi_TD
#
# description:
#   Computes the second derivative of the
#   line-search function phi for TDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - second derivative of phi
##
def dd_phi_TD(alpha, P, T, knots, degree, X):
    D     = -dP_f_TD(P, T, knots, degree, X)
    ddphi = 0.0
    m = len(T)
    for k in range(m):
        n_k = cv.unit_normal(P, T[k], knots, degree)
        d_k = np.dot(n_k, cv.bspline_curve(D, T[k], knots, degree))
        ddphi += d_k * d_k
    return ddphi


##
# function: gradient_descent_TD
#
# description:
#   Performs gradient descent optimization for
#   the Tangent Distance Minimization (TDM).
#
#   At each iteration:
#   - the gradient is computed
#   - an optimal step size alpha is estimated
#     using Newton's method
#   - the control points are updated
#   - the footpoint parameters tk are recomputed
#
# input:
#   - P0 : initial control points
#   - T0 : initial parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum number of iterations
#   - tol : convergence tolerance
#   - constraint : whether endpoint constraints
#                  are enforced
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_TD(P0, T0, knots, degree, X, alpha0=0.1, max_iter=100, tol=1e-6, constraint=True):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, knots, degree, X)))
        grad_P = dP_f_TD(P, T, knots, degree, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_TD, dd_phi_TD, P, T, knots, degree, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        if constraint:
            # constraints can be modified in need
            #P[0] = P0[0]
            #P[-1] = P0[-1]
            P[-1] = P[0]
        T = all_tk(X, P, knots, degree, initial_guesses=T)
    return P, log_iter, log_avg_error


#################################################
#                                               #
#                      SDM                      #
#                                               #
#################################################


##
# function: signed_distance
#
# description:
#   Computes the signed distance between
#   a Bspline curve point P(t) and a data
#   point Xk.
#
#   The sign is determined using the curve
#   normal direction:
#   - positive on one side of the curve
#   - negative on the other side
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - signed distance value
##
def signed_distance(t,Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    d = np.linalg.norm(P_t - Xk)
    n = cv.unit_normal(control_points, t, knots, degree)
    sign = -np.sign(np.dot(P_t - Xk, n))
    return d * sign


##
# function: err_SD
#
# description:
#   Computes the Squared Distance Minimization (SDM)
#   local error associated with data point Xk.
#
#   Two cases are considered:
#
#   - if the squared distance is negative,
#     the full SDM formulation is used
#
#   - if the squared distance is positive
#     and smaller than the curvature radius,
#     the method falls back to the tangent
#     distance formulation
#
# input:
#   - t : parameter value
#   - Xk : data point
#   - control_points : B-spline control points
#   - knots : knot vector
#   - degree : spline degree
#
# output:
#   - SDM local error value
##
def err_SD(t, Xk, control_points, knots, degree):
    d = signed_distance(t,Xk, control_points, knots, degree)
    # SDM
    if (d < 0):
        rho = cv.curvature_radius(control_points, t, knots, degree)
        unit_N = cv.unit_normal(control_points, t, knots, degree)
        unit_T = cv.unit_tangent(control_points, t, knots, degree)
        P_t = cv.bspline_curve(control_points, t, knots, degree)

        coeff = d / (d - rho)
        diff = P_t - Xk
        first_part = coeff * np.dot(diff, unit_T)**2
        second_part = np.dot(diff, unit_N)**2
        return first_part + second_part
    # TDM
    else: # 0 <= d < rho
        return err_TD(t, Xk, control_points, knots, degree)


##
# function: f_SD
#
# description:
#   Computes the global Squared Distance Minimization
#   (SDM) objective function.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - SDM value
##
def f_SD(P, T, knots, degree, X):
    return 0.5 * sum(err_SD(T[k], X[k], P, knots, degree) for k in range(len(T)))


##
# function: dP_f_SD
#
# description:
#   Computes the gradient of the SDM objective
#   function with respect to the control points.
#
# input:
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_SD(P, T, knots, degree, X):
    m = P.shape[0]
    n = len(T)
    grad_P = np.zeros_like(P)

    for k in range(n):
        signed_d = signed_distance(T[k],X[k], P, knots, degree)
        unit_N = cv.unit_normal(P,T[k], knots, degree)
        diff = cv.bspline_curve(P, T[k], knots, degree) - X[k]
        d_projN = np.dot(unit_N, diff)
        if (signed_d < 0):
            rho = cv.curvature_radius(P, T[k], knots, degree)
            coeff = signed_d / (signed_d - rho)
            unit_T = cv.unit_tangent(P,T[k], knots, degree)
            d_projT = np.dot(unit_T, diff)
            for i in range(m):
                B_ik = cv.bspline_basis(i, degree, T[k], knots)
                grad_P[i] += coeff * d_projT * B_ik * unit_T
        for i in range(m):
            B_ik = cv.bspline_basis(i, degree, T[k], knots)
            grad_P[i] += d_projN * B_ik * unit_N
    return grad_P


"""
##
# function: phi_SD
#
# description:
#   Evaluates the line-search function phi
#   used in the Squared Distance Minimization (SDM).
#   This function is not explicitly used, but
#   its derivatives are required for the
#   Newton line-search procedure.
##
def phi_SD(alpha, P, T, knots, degree, X):
    D = -dP_f_SD(P, T, knots, degree, X)
    return f_SD(P + alpha * D, T, knots, degree, X)
"""


##
# function: d_phi_SD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for SDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_SD(alpha, P, T, knots, degree, X):
    D    = -dP_f_SD(P, T, knots, degree, X)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        diff = cv.bspline_curve(P + alpha * D, T[k], knots, degree) - X[k]
        signed_d = signed_distance(T[k],X[k], P, knots, degree)
        if (signed_d < 0):
            rho = cv.curvature_radius(P, T[k], knots, degree)
            unit_T = cv.unit_tangent(P,T[k], knots, degree)
            coeff = signed_d / (signed_d - rho)

            r1 = coeff * np.dot(unit_T , diff)
            d1 = np.dot(unit_T , cv.bspline_curve(D, T[k], knots, degree))
            first_part = r1 * d1
            dphi += first_part
        
        unit_N = cv.unit_normal(P, T[k], knots, degree)
        r = np.dot(unit_N, diff)
        d = np.dot(unit_N, cv.bspline_curve(D, T[k], knots, degree))
        dphi += r * d
    return dphi


##
# function: dd_phi_SD
#
# description:
#   Computes the second derivative of the
#   line-search function phi for SDM.
#
# input:
#   - alpha : step size
#   - P : B-spline control points
#   - T : parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#
# output:
#   - second derivative of phi
##
def dd_phi_SD(alpha, P, T, knots, degree, X):
    D     = -dP_f_SD(P, T, knots, degree, X)
    ddphi = 0.0
    m = len(T)
    for k in range(m):

        signed_d = signed_distance(T[k],X[k], P, knots, degree)
        
        if (signed_d < 0):
            rho = cv.curvature_radius(P, T[k], knots, degree)
            unit_T = cv.unit_tangent(P, T[k], knots, degree)
            coeff = signed_d / (signed_d - rho)

            d = np.dot(unit_T , cv.bspline_curve(D, T[k], knots, degree))
            ddphi += coeff * (d**2)

        unit_N = cv.unit_normal(P, T[k], knots, degree)
        d_k = np.dot(unit_N, cv.bspline_curve(D, T[k], knots, degree))
        ddphi += d_k**2
    return ddphi


##
# function: gradient_descent_SD
#
# description:
#   Performs gradient descent optimization for
#   the Squared Distance Minimization (SDM).
#
#   At each iteration:
#   - the gradient is computed
#   - an optimal step size alpha is estimated
#     using Newton's method
#   - the control points are updated
#   - the footpoint parameters tk are recomputed
#
# input:
#   - P0 : initial control points
#   - T0 : initial parameter values
#   - knots : knot vector
#   - degree : spline degree
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum iterations
#   - tol : convergence tolerance
#   - constraint : whether endpoint constraints
#                  are enforced
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_SD(P0, T0, knots, degree, X, alpha0=0.1, max_iter=100, tol=1e-6, constraint=True):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, knots, degree, X)))
        grad_P = dP_f_SD(P, T, knots, degree, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_SD, dd_phi_SD, P, T, knots, degree, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        if constraint:
            # constraints can be modified in need
            #P[0] = P0[0]
            #P[-1] = P0[-1]
            P[-1] = P[0]
        T = all_tk(X, P, knots, degree, initial_guesses=T)
    return P, log_iter, log_avg_error