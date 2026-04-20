import numpy as np
from . import bspline_curves as cv

# function to minimize for finding tk
def F(control_points, t, Xk):
    P = cv.bspline_curve(control_points, t)
    return np.linalg.norm(P - Xk) ** 2

# function for which we want to find the 0
def F_prime(control_points,t, Xk):
    P = cv.bspline_curve(control_points, t)
    Pprime = cv.dt_bspline_curve(control_points, t)
    return 2 * np.dot(P - Xk, Pprime)

def F_primprim(control_points, t, Xk):
    P = cv.bspline_curve(control_points, t)
    Pprime = cv.dt_bspline_curve(control_points, t)
    Pprimprim = cv.dtt_bspline_curve(control_points, t)
    return 2 * np.dot(Pprime, Pprime) + 2 * np.dot(P - Xk, Pprimprim)

def newton_tk(Xk, control_points, initial_guess, tol=1e-6, max_iter=100):
    tk = initial_guess
    #t_min = knots[degree]
    #t_max = knots[-degree - 1]
    #tk = np.clip(initial_guess, t_min, t_max)  # we can clip tk to keep it in a valid range
    for _ in range(max_iter):
        Fp = F_prime(control_points, tk, Xk)
        Fpp = F_primprim(control_points, tk, Xk)
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

# objective function
def f(P, T):
    tot = 0.0
    for k in range(len(T)):
        tot += np.linalg.norm((cv.bspline_curve(P, T[k]) - X[k])) ** 2
    return 0.5 * tot
 
def dt_f(P, T):
    n = len(T)
    grad_t = np.zeros(n)
    for k in range(n):   
        grad_t[k] = np.dot(cv.bspline_curve(P, T[k]) - X[k], cv.dt_bspline_curve(P, T[k]))
    return grad_t

def dP_f(P, T):
    m = P.shape[0]
    n = len(T)
    grad_P = np.zeros_like(P)

    for k in range(n):
        d = cv.bspline_curve(P, T[k]) - X[k]
        for i in range(m):
            B_ik = cv.bspline_basis(i, degree, T[k])
            grad_P[i] += d * B_ik
    return grad_P

def phi(alpha, P, T):
    D = -dP_f(P, T)
    return f(P + alpha * D, T)

def d_phi(alpha, P, T):
    D    = -dP_f(P, T)
    dphi = 0.0
    m = len(T)
    for k in range(m):
        r_k = cv.bspline_curve(P + alpha * D, T[k]) - X[k]
        d_k = cv.bspline_curve(D, T[k])
        dphi += np.dot(r_k, d_k)
    return dphi

def dd_phi(alpha, P, T):
    D     = -dP_f(P, T)
    ddphi = 0.0
    m = len(T)
    for k in range(m):
        d_k = cv.bspline_curve(D, T[k])
        ddphi += np.dot(d_k, d_k)
    return ddphi


def newton_alpha(f, grad_f, P, T, alpha0, tol=1e-6, max_iter=1000): # changer f et grad_f pour d_phi et dd_phi
    alpha = alpha0
    for _ in range(max_iter):
        if np.linalg.norm(grad_f(alpha, P, T)) < tol:
            break
        alpha = alpha - f(alpha, P, T) / grad_f(alpha, P, T)
    return alpha



def gradient_descent(x_0, f, df, alpha0, T, max_iter=1000, tol=1e-6):
    x = x_0
    alpha = alpha0
    for i in range(max_iter): 
        fx = f(alpha0, x, T)
        dfx = df(alpha0, x, T)
        alpha = newton_alpha(f, df, x, T, alpha0, tol)
        dx = - alpha * dfx
        alpha0 = alpha
        if np.linalg.norm(dx) < tol:
            break
        x += dx
    return x


# variables définis dans le main
degree = 2
knots = np.array([0,0,0,1,1,1])
X = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])
Pc = np.array( # points de contrôle, initial guess
    [
        [0  ,   0],
        [0.4,   0],
        [1.0,   0]
    ])

initial_guess = 0.1
tkk = newton_tk(X[1], Pc, initial_guess, tol=1e-6, max_iter=1000)
T = [0.0, tkk, 1.0]

#alphaa = newton_alpha(d_phi, dd_phi, Pc, T, alpha0=0.1, tol=1e-6, max_iter=1000)
#alpha0 = 0.1

#print(tkk)
#print(alphaa)

#P_plus = gradient_descent(Pc, phi, d_phi, alpha0, T, max_iter=1000, tol=1e-6)
#print(P_plus)