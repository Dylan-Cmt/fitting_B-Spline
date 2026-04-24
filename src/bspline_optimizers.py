import numpy as np
from . import bspline_curves as cv

"""
# function to minimize for finding tk
def F(control_points, t, Xk):
    P = cv.bspline_curve(control_points, t)
    return np.linalg.norm(P - Xk) ** 2
"""


# function for which we want to find the 0
def F_prime(t, Xk, control_points, knots, degree):
    P = cv.bspline_curve(control_points, t, knots, degree)
    Pprime = cv.dt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(P - Xk, Pprime)


def F_primprim(t, Xk, control_points, knots, degree):
    P = cv.bspline_curve(control_points, t, knots, degree)
    Pprime = cv.dt_bspline_curve(control_points, t, knots, degree)
    Pprimprim = cv.dtt_bspline_curve(control_points, t, knots, degree)
    return 2 * np.dot(Pprime, Pprime) + 2 * np.dot(P - Xk, Pprimprim)


def newton_tk(Xk, control_points, knots, degree, initial_guess, tol=1e-6, max_iter=100):
    t = np.clip(initial_guess, 0.0, 1.0) # keep tk in [0;1]
    for _ in range(max_iter):
        Fp = F_prime(t, Xk, control_points, knots, degree)
        Fpp = F_primprim(t, Xk, control_points, knots, degree)
        if Fpp == 0:
            break
        t_new = t - Fp / Fpp
        t_new = np.clip(t_new, 0.0, 1.0)
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
    return t

def all_tk(X, control_points, knots, degree, initial_guesses=None):
    X = np.asarray(X, dtype=float)
    if initial_guesses is None:
        initial_guesses = np.linspace(0.0, 1.0, len(X))
    return [newton_tk(Xk, control_points, knots, degree, guess) for Xk, guess in zip(X, initial_guesses)]



# objective function
def f(P, T, knots, degree, X):
    tot = 0.0
    for k in range(len(T)):
        tot += np.linalg.norm((cv.bspline_curve(P, T[k], knots, degree) - X[k])) ** 2
    return 0.5 * tot


def dt_f(P, T, knots, degree, X):
    n = len(T)
    grad_t = np.zeros(n)
    for k in range(n):   
        grad_t[k] = np.dot(cv.bspline_curve(P, T[k], knots, degree) - X[k], cv.dt_bspline_curve(P, T[k], knots, degree))
    return grad_t


def dP_f(P, T, knots, degree, X):
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
def phi(alpha, P, T, X):
    D = -dP_f(P, T, knots, degree, X)
    return f(P + alpha * D, T, knots, degree)
"""


def d_phi(alpha, P, T, knots, degree, X):
    D    = -dP_f(P, T, knots, degree, X)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        r_k = cv.bspline_curve(P + alpha * D, T[k], knots, degree) - X[k]
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        dphi += np.dot(r_k, d_k)
    return dphi


def dd_phi(alpha, P, T, knots, degree, X):
    D     = -dP_f(P, T, knots, degree, X)
    ddphi = 0.0
    m = len(T)
    for k in range(m):
        d_k = cv.bspline_curve(D, T[k], knots, degree)
        ddphi += np.dot(d_k, d_k)
    return ddphi


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


def avg_error(P, T, knots, degree, X):
    return np.sqrt(2 * f(P, T, knots, degree, X) / len(X))


def gradient_descent(P0, T0, knots, degree, X, alpha0=0.1, max_iter=100, tol=1e-6):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, knots, degree, X)))
        grad_P = dP_f(P, T, knots, degree, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi, dd_phi, P, T, knots, degree, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        P[0] = P0[0]
        P[-1] = P0[-1]
        T = all_tk(X, P, knots, degree, initial_guesses=T)
    return P, log_iter, log_avg_error