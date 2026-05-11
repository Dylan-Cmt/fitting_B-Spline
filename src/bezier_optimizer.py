import numpy as np
from . import bezier_curves as cv


#################################################
#                                               #
#           Basis of all methods                #
#                                               #
#################################################

# function to minimize for finding tk
def err_PD(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    return np.linalg.norm(P_t - Xk) ** 2


def err_PD_prime(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    P_t_prime = cv.eval_bezier_derivative(control_points, t)
    return 2 * np.dot(P_t - Xk, P_t_prime)


def err_PD_primprim(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    P_t_prime = cv.eval_bezier_derivative(control_points, t)
    P_t_second = cv.eval_bezier_second_derivative(control_points, t)
    return 2 * np.dot(P_t_prime, P_t_prime) + 2 * np.dot(P_t - Xk, P_t_second)


def newton_tk(Xk, control_points, initial_guess, tol=1e-6, max_iter=100):
    t = np.clip(initial_guess, 0.0, 1.0) # keep tk in [0;1]
    for _ in range(max_iter):
        Fp = err_PD_prime(t, Xk, control_points)
        Fpp = err_PD_primprim(t, Xk, control_points)
        if abs(Fpp) < 1e-14:
            break
        t_new = t - Fp / Fpp
        t_new = np.clip(t_new, 0.0, 1.0)
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
    return t



def all_tk(X, control_points, initial_guesses=None, n_samples=50):
    # for the first iter the first initial guess is found automatically
    if initial_guesses is None: 
        t_samples = np.linspace(0, 1, n_samples)
        initial_guesses = []
        for Xk in X:
            dists = [err_PD(t, Xk, control_points) for t in t_samples]
            initial_guesses.append(t_samples[np.argmin(dists)])
    # for the next iter, we take previous tks as initial guesses
    return [newton_tk(Xk, control_points, t0)
            for Xk, t0 in zip(X, initial_guesses)]

# d_phi and dd_phi are defined below
# alpha is computed for gradient descents below
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


# f_PD is defined directly below
def avg_error(P, T, X):
    return np.sqrt(2 * f_PD(P, T, X) / len(X))

#################################################
#                                               #
#                      PDM                      #
#                                               #
#################################################


def f_PD(P, T, X):
    total = 0.0
    for tk, Xk in zip(T, X):
        diff = cv.eval_bezier_curve(P, tk) - Xk
        total += np.dot(diff, diff)
    return 0.5 * total


def dP_f_PD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        grad += np.outer(basis, diff)
    return grad

"""
def phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    return f_PD(P + alpha * D, T, X)
"""

def d_phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        r = cv.eval_bezier_curve(P + alpha * D, tk) - Xk
        d = cv.eval_bezier_curve(D, tk)
        dphi += np.dot(r, d)
    return dphi


def dd_phi_PD(alpha, P, T, X):
    D = -dP_f_PD(P, T, X)
    ddphi = 0.0
    for tk in T:
        d = cv.eval_bezier_curve(D, tk)
        ddphi += np.dot(d, d)
    return ddphi


def avg_error_PD(P, T, X):
    return np.sqrt(2 * f_PD(P, T, X) / len(X))

def gradient_descent_PD(P0, T0, X, alpha0=0.1, max_iter=100, tol=1e-6):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error_PD(P, T, X)))
        grad_P = dP_f_PD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_PD, dd_phi_PD, P, T, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        P[0] = P0[0]
        P[-1] = P0[-1]
        T = all_tk(X, P, initial_guesses=T)
    return P, log_iter, log_avg_error


#################################################
#                                               #
#                      TDM                      #
#                                               #
#################################################


def err_TD(t, Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    return np.dot((P_t - Xk), cv.unit_normal(control_points,t))**2
    

def f_TD(P, T, X):
    total = 0.0
    for tk, Xk in zip(T, X):
        total += err_TD(tk, Xk, P)
    return 0.5 * total


def dP_f_TD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        unit_N = cv.unit_normal(P,tk)
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        grad += np.outer(basis, unit_N * np.dot(unit_N, diff))
    return grad

"""
def phi(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    return f_TD(P + alpha * D, T, X)
"""

def d_phi_TD(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        unit_N = cv.unit_normal(P,tk)
        r = np.dot(unit_N , cv.eval_bezier_curve(P + alpha * D, tk) - Xk)
        d = np.dot(unit_N , cv.eval_bezier_curve(D, tk))
        dphi += r * d
    return dphi


def dd_phi_TD(alpha, P, T, X):
    D = -dP_f_TD(P, T, X)
    ddphi = 0.0
    for tk in T:
        unit_N = cv.unit_normal(P,tk)
        d = np.dot(unit_N , cv.eval_bezier_curve(D, tk))
        ddphi += d**2
    return ddphi


def gradient_descent_TD(P0, T0, X, alpha0=0.1, max_iter=100, tol=1e-6):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f_TD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_TD, dd_phi_TD, P, T, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        P[0] = P0[0]
        P[-1] = P0[-1]
        T = all_tk(X, P, initial_guesses=T)
    return P, log_iter, log_avg_error


#################################################
#                                               #
#                      SDM                      #
#                                               #
#################################################

def signed_distance(t,Xk, control_points):
    P_t = cv.eval_bezier_curve(control_points, t)
    d = np.linalg.norm(P_t - Xk)
    n = cv.unit_normal(control_points, t)
    sign = -np.sign(np.dot(P_t - Xk, n))
    return d * sign


def err_SD(t, Xk, control_points):
    d = signed_distance(t,Xk, control_points)
    roh = cv.curvature_radius(control_points, t)
    if (d < 0):
        T = cv.unit_tangent(control_points, t)
        N = cv.unit_normal(control_points, t)
        P_t = cv.eval_bezier_curve(control_points, t)

        coeff = d / (d - roh)
        diff = P_t - Xk
        first_part = coeff * np.dot(diff, T)**2
        second_part = np.dot(diff, N)**2
        return first_part + second_part
    elif (0 <= d < roh):
        return err_TD(t, Xk, control_points)
    else:
        raise ValueError("d must not be higher than roh, footpoint calculus failed")

    

def f_SD(P, T, X):
    return 0.5 * sum(err_SD(T[k], X[k], P) for k in range(len(T)))


def dP_f_SD(P, T, X):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    grad = np.zeros_like(P)
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk,Xk, P)
        unit_N = cv.unit_normal(P,tk)
        diff = cv.eval_bezier_curve(P, tk) - Xk
        basis = cv.bernstein_basis_vector(n, tk)
        if (signed_d < 0):
            roh = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P,tk)
            coeff = signed_d / (signed_d - roh)

            first_part = coeff * np.outer(basis, unit_T * np.dot(unit_T, diff))
            second_part = np.outer(basis, unit_N * np.dot(unit_N, diff))
            grad += first_part

        grad += np.outer(basis, unit_N * np.dot(unit_N, diff))
    return grad

"""
def phi(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)
    return f_SD(P + alpha * D, T, X)
"""

def d_phi_SD(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)
    dphi = 0.0
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk,Xk, P)
        diff = cv.eval_bezier_curve(P + alpha * D, tk) - Xk
        if (signed_d < 0):
            roh = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P,tk)
            coeff = signed_d / (signed_d - roh)

            r1 = coeff * np.dot(unit_T , diff)
            d1 = np.dot(unit_T , cv.eval_bezier_curve(D, tk))
            first_part = r1 * d1
            dphi += first_part
        
        unit_N = cv.unit_normal(P,tk)
        r = np.dot(unit_N , diff)
        d = np.dot(unit_N , cv.eval_bezier_curve(D, tk))
        dphi += r * d
    return dphi


def dd_phi_SD(alpha, P, T, X):
    D = -dP_f_SD(P, T, X)
    ddphi = 0.0
    for tk, Xk in zip(T, X):
        signed_d = signed_distance(tk,Xk, P)
        
        if (signed_d < 0):
            roh = cv.curvature_radius(P, tk)
            unit_T = cv.unit_tangent(P,tk)
            coeff = signed_d / (signed_d - roh)

            d = np.dot(unit_T , cv.eval_bezier_curve(D, tk))
            ddphi += coeff * (d**2)

        unit_N = cv.unit_normal(P,tk)
        d = np.dot(unit_N , cv.eval_bezier_curve(D, tk))
        ddphi += d**2
    return ddphi


def gradient_descent_SD(P0, T0, X, alpha0=0.1, max_iter=100, tol=1e-6):
    P = np.asarray(P0, dtype=float).copy()
    T = np.asarray(T0, dtype=float).copy()
    log_avg_error = []
    log_iter = []
    for i in range(max_iter):
        log_iter.append(np.log10(i+1))
        log_avg_error.append(np.log10(avg_error(P, T, X)))
        grad_P = dP_f_SD(P, T, X)
        norm_grad = np.linalg.norm(grad_P)
        if norm_grad < tol:
            print(f"Convergence achieved after {i+1} iterations")
            break
        D = -grad_P
        alpha = newton_alpha(d_phi_SD, dd_phi_SD, P, T, X, alpha0, tol)
        if alpha <= 0 or np.isnan(alpha):
            alpha = alpha0
        P += alpha * D
        P[0] = P0[0]
        P[-1] = P0[-1]
        T = all_tk(X, P, initial_guesses=T)
    return P, log_iter, log_avg_error
