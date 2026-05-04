import numpy as np
from . import bspline_curves as cv


#################################################
#                                               #
#           Basis of all methods                #
#                                               #
#################################################


# function to minimize for finding tk
def err_PD(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    return np.linalg.norm(P_t - Xk) ** 2


# function for which we want to find the 0
def err_PD_prime(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    P_t_prime = cv.dt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(P_t - Xk, P_t_prime)


def err_PD_primprim(t, Xk, control_points, knots, degree):
    P_t = cv.bspline_curve(control_points, t, knots, degree)
    P_t_prime = cv.dt_bspline_curve(control_points, t, knots, degree)
    P_t_second = cv.dtt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(P_t_prime, P_t_prime) + 2 * np.dot(P_t - Xk, P_t_second)


def newton_tk(Xk, control_points, knots, degree, initial_guess,
              tol=1e-6, max_iter=100):
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


def all_tk(X, control_points, knots, degree, initial_guesses=None, n_samples=50):
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    
    # for the first iter the first initial guess is found automatically
    if initial_guesses is None: 
        t_samples = np.linspace(t_min, t_max, n_samples, endpoint=False)
        initial_guesses = []
        for Xk in X:
            dists = [err_PD(t, Xk, control_points, knots, degree) for t in t_samples]
            initial_guesses.append(t_samples[np.argmin(dists)])
    # for the next iter, we take previous tks as initial guesses
    return [newton_tk(Xk, control_points, knots, degree, t0)
            for Xk, t0 in zip(X, initial_guesses)]


# d_phi and dd_phi are defined below
# alpha is computed for gradient descents below
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


# f_PD is defined directly below
def avg_error(P, T, knots, degree, X):
    return np.sqrt(2 * f_PD(P, T, knots, degree, X) / len(X))


#################################################
#                                               #
#                      PDM                      #
#                                               #
#################################################

# objective function
def f_PD(P, T, knots, degree, X):
    return 0.5 * sum(err_PD(T[k], X[k], P, knots, degree) for k in range(len(T)))


"""
def dt_f_PD(P, T, knots, degree, X):
    n = len(T)
    grad_t = np.zeros(n)
    for k in range(n):   
        grad_t[k] = np.dot(cv.bspline_curve(P, T[k], knots, degree) - X[k], cv.dt_bspline_curve(P, T[k], knots, degree))
    return grad_t
"""


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
def phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, knots, degree, X)
    return f_PD(P + alpha * D, T, knots, degree)
"""


def d_phi_PD(alpha, P, T, knots, degree, X):
    D    = -dP_f_PD(P, T, knots, degree, X)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        r_k = cv.bspline_curve(P + alpha * D, T[k], knots, degree) - X[k]
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        dphi += np.dot(r_k, d_k)
    return dphi


def dd_phi_PD(alpha, P, T, knots, degree, X):
    D     = -dP_f_PD(P, T, knots, degree, X)
    ddphi = 0.0
    m = len(T)
    for k in range(m):
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        ddphi += np.dot(d_k, d_k)
    return ddphi


def gradient_descent_PD(P0, T0, knots, degree, X, alpha0=0.1, max_iter=100, tol=1e-6):
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
        P[0] = P0[0]
        P[-1] = P0[-1]
        T = all_tk(X, P, knots, degree, initial_guesses=T)
    return P, log_iter, log_avg_error