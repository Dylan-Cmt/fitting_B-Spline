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

"""
def F(t, Xk, control_points):
    P_t = eval_bezier_curve(control_points, t)
    return np.linalg.norm(P_t - Xk) ** 2
"""

def F_prime(t, Xk, control_points):
    P_t = eval_bezier_curve(control_points, t)
    P_t_prime = eval_bezier_derivative(control_points, t)
    return 2 * np.dot(P_t - Xk, P_t_prime)


def F_primprim(t, Xk, control_points):
    P_t = eval_bezier_curve(control_points, t)
    P_t_prime = eval_bezier_derivative(control_points, t)
    P_t_second = eval_bezier_second_derivative(control_points, t)
    return 2 * np.dot(P_t_prime, P_t_prime) + 2 * np.dot(P_t - Xk, P_t_second)


def newton_tk(Xk, control_points, initial_guess, tol=1e-6, max_iter=100):
    t = np.clip(initial_guess, 0.0, 1.0) # keep tk in [0;1]
    for _ in range(max_iter):
        Fp = F_prime(t, Xk, control_points)
        Fpp = F_primprim(t, Xk, control_points)
        if Fpp == 0:
            break
        t_new = t - Fp / Fpp
        t_new = np.clip(t_new, 0.0, 1.0)
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
    return t


def all_tk(X, control_points, initial_guesses=None):
    X = np.asarray(X, dtype=float)
    if initial_guesses is None:
        initial_guesses = np.linspace(0.0, 1.0, len(X))
    return [newton_tk(Xk, control_points, guess) for Xk, guess in zip(X, initial_guesses)]


def f(P, T, X):
    total = 0.0
    for tk, Xk in zip(T, X):
        diff = eval_bezier_curve(P, tk) - Xk
        total += np.dot(diff, diff)
    return 0.5 * total


def dP_f(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        diff = eval_bezier_curve(P, tk) - Xk
        basis = bernstein_basis_vector(n, tk)
        grad += np.outer(basis, diff)
    return grad

"""
def phi(alpha, P, T, X):
    D = -dP_f(P, T, X)
    return f(P + alpha * D, T, X)
"""

def d_phi(alpha, P, T, X):
    D = -dP_f(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        r = eval_bezier_curve(P + alpha * D, tk) - Xk
        d = eval_bezier_curve(D, tk)
        dphi += np.dot(r, d)
    return dphi


def dd_phi(alpha, P, T, X):
    D = -dP_f(P, T, X)
    ddphi = 0.0
    for tk in T:
        d = eval_bezier_curve(D, tk)
        ddphi += np.dot(d, d)
    return ddphi


def newton_alpha(d_phi_func, dd_phi_func, P, T, X, alpha0, tol=1e-6, max_iter=100):
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

def avg_error(P, T, X):
    return np.sqrt(2 * f(P, T, X) / len(X))

def gradient_descent(P0, T, X, alpha0=0.1, max_iter=100, tol=1e-6):
    P = np.asarray(P0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi, dd_phi, P, T, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
    return P, log_iter, log_avg_error
    