import numpy as np
from . import bezier_curves as cv


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
#   between a Bézier curve point P(t)
#   and a data point Xk.
#
# input:
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - squared point distance error
##
def err_PD(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    return np.linalg.norm(P_t - Xk) ** 2


##
# function: err_PD_prime
#
# description:
#   Computes the first derivative of the
#   point distance error with respect to t.
#
# input:
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - first derivative of the error
##
def err_PD_prim(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    P_t_prime = cv.eval_bezier_derivative(control_points, t)
    return 2 * np.dot(P_t - Xk, P_t_prime)


##
# function: err_PD_primprim
#
# description:
#   Computes the second derivative of the
#   point distance error with respect to t.
#
# input:
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - second derivative of the error
##
def err_PD_primprim(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    P_t_prime = cv.eval_bezier_derivative(control_points, t)
    P_t_second = cv.eval_bezier_second_derivative(control_points, t)
    return 2 * np.dot(P_t_prime, P_t_prime) + 2 * np.dot(P_t - Xk, P_t_second)


##
# function: newton_tk
#
# description:
#   Uses Newton's method to compute the foot point of Xk.
#   A faster two-phase approach (preprocessing and query)
#   could be employed for improved convergence, but
#   Newton's method is preferred here.
#
# input:
#   - err_prim : function returning f'(t)
#   - err_primprim : function returning f''(t)
#   - Xk : data point
#   - control_points : Bézier control points
#   - initial_guess : initial value for tk
#   - tol : convergence tolerance
#   - max_iter : maximum number of iterations
#
# output:
#   - optimized parameter tk
##
def newton_tk(err_prim, err_primprim, Xk,control_points,initial_guess,tol=1e-6,max_iter=100):
    t = np.clip(initial_guess, 0.0, 1.0)

    for _ in range(max_iter):
        Fp = err_prim(t, Xk, control_points)
        Fpp = err_primprim(t, Xk, control_points)
        if abs(Fpp) < 1e-14:
            break
        t_new = t - Fp / Fpp
        t_new = np.clip(t_new, 0.0, 1.0)
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
#   - X : array of data points
#   - control_points : Bézier control points
#   - initial_guesses : optional initial tk values
#   - n_samples : number of samples for initialization
#
# output:
#   - list of optimized foot points tk for every Xk
##
def all_tk(X,control_points,initial_guesses=None,n_samples=50):
    # First iteration:
    # initial guesses are estimated automatically
    if initial_guesses is None:

        t_samples = np.linspace(0, 1, n_samples)
        initial_guesses = []
        for Xk in X:
            dists = [err_PD(t, Xk, control_points) for t in t_samples]
            initial_guesses.append(t_samples[np.argmin(dists)])

    # Next iterations:
    # previous tk values are reused
    return [
        newton_tk(err_PD_prim, err_PD_primprim, Xk, control_points, t0)
        for Xk, t0 in zip(X, initial_guesses)
    ]


##
# function: newton_alpha
#
# description:
#   Uses Newton's method to compute the optimal
#   step size alpha for gradient descent.
#
# input:
#   - d_phi_func : first derivative of phi
#   - dd_phi_func : second derivative of phi
#   - P : control points
#   - T : parameter values
#   - X : data points
#   - alpha0 : initial alpha value
#   - tol : convergence tolerance
#   - max_iter : maximum number of iterations
#
# output:
#   - optimized alpha value
##
def newton_alpha(d_phi_func,dd_phi_func,P,T,X,alpha0,tol=1e-6,max_iter=100):
    alpha = alpha0
    for _ in range(max_iter):
        grad = d_phi_func(alpha, P, T, X)
        if abs(grad) < tol:
            break
        denom = dd_phi_func(alpha, P, T, X)
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
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - average fitting error
##
def avg_error(P, T, X):
    return np.sqrt(2 * f_PD(P, T, X) / len(X))

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
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - PDM value
##
def f_PD(P, T, X):
    total = 0.0
    for tk, Xk in zip(T, X):
        diff = cv.eval_bezier_curve(P, tk) - Xk
        total += np.dot(diff, diff)
    return 0.5 * total


##
# function: dP_f_PD
#
# description:
#   Computes the gradient of the PDM objective
#   function with respect to the control points.
#
# input:
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_PD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        grad += np.outer(basis, diff)
    return grad

##
# function: phi_PD
#
# description:
#   Evaluates the line-search function phi
#   used in the Point Distance Minimization (PDM).
#   This function is not explicitly used here, but
#   its derivatives are required for the
#   Newton line-search procedure.
#   The function is used in the testing file
##
def phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    return f_PD(P + alpha * D, T, X)


##
# function: d_phi_PD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for PDM.
#
# input:
#   - alpha : step size
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        r = cv.eval_bezier_curve(P + alpha * D, tk) - Xk
        d = cv.eval_bezier_curve(D, tk)
        dphi += np.dot(r, d)
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
#   - P : control points
#   - T : parameter values
#
# output:
#   - second derivative of phi
##
def dd_phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    ddphi = 0.0
    for tk in T:
        d = cv.eval_bezier_curve(D, tk)
        ddphi += np.dot(d, d)
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
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum number of iterations
#   - tol : convergence tolerance
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_PD(P0,T0,X, alpha0=0.1,max_iter=100,tol=1e-6,constraint="opened"):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i + 1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f_PD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        # Convergence test
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        # Descent direction
        D = -grad_P
        # Newton line-search
        alpha = newton_alpha(d_phi_PD,dd_phi_PD,P,T,X,alpha0,tol)
        # Fallback step size
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        # Update control points
        P += alpha * D
        # constraints can be modified in need
        if constraint=="opened":
            P[0] = P0[0]
            P[-1] = P0[-1]
        elif constraint=="closed":
            P[-1] = P[0]
        # Recompute footpoint parameters
        T = all_tk(X, P, initial_guesses=T)
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
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - tangent distance error
##
def err_TD(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)

    return np.dot((P_t - Xk),cv.unit_normal(control_points, t)) ** 2


##
# function: f_TD
#
# description:
#   Computes the global Tangent Distance Minimization
#   (TDM) objective function.
#
# input:
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - TDM value
##
def f_TD(P, T, X):
    total = 0.0
    for tk, Xk in zip(T, X):
        total += err_TD(tk, Xk, P)
    return 0.5 * total


##
# function: dP_f_TD
#
# description:
#   Computes the gradient of the TDM objective
#   function with respect to the control points.
#
# input:
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_TD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        unit_N = cv.unit_normal(P, tk)
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        grad += np.outer(basis,unit_N * np.dot(unit_N, diff))
    return grad


##
# function: phi_TD
#
# description:
#   Evaluates the line-search function phi
#   used in the Tangent Distance Minimization (TDM).
#   This function is not explicitly used here, but
#   its derivatives are required for the
#   Newton line-search procedure.
#   The function is used in the testing file
##
def phi_TD(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    return f_TD(P + alpha * D, T, X)



##
# function: d_phi_TD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for TDM.
#
# input:
#   - alpha : step size
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_TD(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        unit_N = cv.unit_normal(P , tk)
        r = np.dot(unit_N , cv.eval_bezier_curve(P + alpha * D, tk) - Xk)
        d = np.dot(unit_N , cv.eval_bezier_curve(D, tk))
        dphi += r * d
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
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - second derivative of phi
##
def dd_phi_TD(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    ddphi = 0.0
    for tk in T:
        unit_N = cv.unit_normal(P, tk)
        d = np.dot(unit_N,cv.eval_bezier_curve(D, tk))
        ddphi += d ** 2
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
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum number of iterations
#   - tol : convergence tolerance
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_TD(P0,T0,X,alpha0=0.1,max_iter=100,tol=1e-6,constraint="opened"):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i + 1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f_TD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        # Convergence test
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        # Descent direction
        D = -grad_P
        # Newton line-search
        alpha = newton_alpha(d_phi_TD,dd_phi_TD,P,T,X,alpha0,tol)
        # Fallback step size
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        # Update control points
        P += alpha * D
        # constraints can be modified in need
        if constraint=="opened":
            P[0] = P0[0]
            P[-1] = P0[-1]
        elif constraint=="closed":
            P[-1] = P[0]
        # Recompute footpoint parameters
        T = all_tk(X, P, initial_guesses=T)
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
#   a Bézier curve point P(t) and a data
#   point Xk.
#
#   The sign is determined using the curve
#   normal direction:
#   - positive on one side of the curve
#   - negative on the other side
#
# input:
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - signed distance value
##
def signed_distance(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    d = np.linalg.norm(P_t - Xk)
    n = cv.unit_normal(control_points, t)
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
#   - t : parameter in [0,1]
#   - Xk : data point
#   - control_points : Bézier control points
#
# output:
#   - SDM local error value
##
def err_SD(t, Xk, control_points):
    d = signed_distance(t, Xk, control_points)
    rho = cv.curvature_radius(control_points, t)
    # SDM 
    if d < 0:
        T = cv.unit_tangent(control_points, t)
        N = cv.unit_normal(control_points, t)
        P_t = cv.eval_bezier_curve(control_points, t)
        coeff = d / (d - rho)
        diff = P_t - Xk

        first_part = coeff * np.dot(diff, T) ** 2
        second_part = np.dot(diff, N) ** 2

        return first_part + second_part
    # TDM
    else: # 0 <= d < rho
        return err_TD(t, Xk, control_points)


##
# function: f_SD
#
# description:
#   Computes the global Squared Distance Minimization
#   (SDM) objective function.
#
# input:
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - SDM value
##
def f_SD(P, T, X):
    return 0.5 * sum(
        err_SD(T[k], X[k], P)
        for k in range(len(T))
    )


##
# function: dP_f_SD
#
# description:
#   Computes the gradient of the SDM objective
#   function with respect to the control points.
#
# input:
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - gradient matrix
##
def dP_f_SD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk, Xk, P)
        unit_N = cv.unit_normal(P, tk)
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        if signed_d < 0:
            rho = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P, tk)
            coeff = signed_d / (signed_d - rho)
            first_part = coeff * np.outer(basis,unit_T * np.dot(unit_T, diff))
            grad += first_part
        grad += np.outer(basis,unit_N * np.dot(unit_N, diff))
    return grad



##
# function: phi_SD
#
# description:
#   Evaluates the line-search function phi
#   used in the Squared Distance Minimization (SDM).
#   This function is not explicitly used here, but
#   its derivatives are required for the
#   Newton line-search procedure.
#   The function is used in the testing file
##
def phi_SD(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)

    return f_SD(P + alpha * D, T, X)



##
# function: d_phi_SD
#
# description:
#   Computes the first derivative of the
#   line-search function phi for SDM.
#
# input:
#   - alpha : step size
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - first derivative of phi
##
def d_phi_SD(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk, Xk, P)
        diff = (cv.eval_bezier_curve(P + alpha * D, tk)- Xk)
        if signed_d < 0:
            rho = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P, tk)
            coeff = signed_d / (signed_d - rho)
            r1 = coeff * np.dot(unit_T, diff)
            d1 = np.dot(unit_T,cv.eval_bezier_curve(D, tk))
            first_part = r1 * d1
            dphi += first_part
        unit_N = cv.unit_normal(P, tk)
        r = np.dot(unit_N, diff)
        d = np.dot(unit_N,cv.eval_bezier_curve(D, tk))
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
#   - P : control points
#   - T : parameter values
#   - X : data points
#
# output:
#   - second derivative of phi
##
def dd_phi_SD(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)
    ddphi = 0.0
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk, Xk, P)
        if signed_d < 0:
            rho = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P, tk)
            coeff = signed_d / (signed_d - rho)
            d = np.dot(unit_T,cv.eval_bezier_curve(D, tk))
            ddphi += coeff * (d ** 2)
        unit_N = cv.unit_normal(P, tk)
        d = np.dot(unit_N,cv.eval_bezier_curve(D, tk))
        ddphi += d ** 2
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
#   - X : data points
#   - alpha0 : initial step size
#   - max_iter : maximum number of iterations
#   - tol : convergence tolerance
#
# output:
#   - optimized control points
#   - logarithm of iteration indices
#   - logarithm of average fitting errors
##
def gradient_descent_SD(P0,T0,X,alpha0=0.1,max_iter=100,tol=1e-6,constraint="opened"):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i + 1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f_SD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        # Convergence test
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        # Descent direction
        D = -grad_P
        # Newton line-search
        alpha = newton_alpha(d_phi_SD,dd_phi_SD,P,T,X,alpha0,tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        # Update control points
        P += alpha * D
        # constraints can be modified in need
        if constraint=="opened":
            P[0] = P0[0]
            P[-1] = P0[-1]
        elif constraint=="closed":
            P[-1] = P[0]
        # Recompute footpoint parameters
        T = all_tk(X, P, initial_guesses=T)
    return P, log_iter, log_avg_error