import numpy as np
import curves as crv

##############################################################################################
##                                                                                          ##
##                                 gradient descent                                         ##
##                                                                                          ##
##############################################################################################

"""
def f(controle_points, degree, nodes, T, data_points):
    f = 0
    for k in range(len(T)):
        tk = T[k]
        f += np.sum((crv.evalbsplinecurve_at_t(degree, nodes, controle_points, tk) - data_points[k]) ** 2)
    return 0.5 * f

def grad_f(controle_points, degree, nodes, T, data_points):
    n = len(controle_points)
    m = len(data_points)
    dim = controle_points.shape[1]

    grad = np.zeros_like(n)
    for l in range(n):
        for k in range(m):
            tk = T[k]
            for j in range(dim):
                curve_P_at_tk = 0
                for i in range(n):
                    curve_P_at_tk += controle_points[i][j] * crv.evalbspline_basis(i, degree, nodes, tk)
                grad[l] += (curve_P_at_tk - data_points[k][j])
            grad[l] *= crv.evalbspline_basis(l, degree, nodes, tk)
    return grad

def phi(alpha, controle_points, degree, nodes, tk, data_points):
    return f(controle_points - alpha * grad_f(controle_points, degree, nodes, tk, data_points), degree, nodes, tk, data_points)

def phi_prime(alpha , controle_points, degree, nodes, tk, data_points):
    n = len(controle_points)
    m = len(data_points)
    dim = controle_points.shape[1]

    grad_phi = 0
    for l in range(n):
        for k in range(m):
            for j in range(dim):
                curve_P_at_tk = 0
                for i in range(n):
                    curve_P_at_tk += controle_points[i][j] * crv.evalbspline_basis(i, degree, nodes, tk)
                grad_phi += (curve_P_at_tk - data_points[k][j]) * crv.evalbspline_basis(l, degree, nodes, tk)
    return 


def newton_alpha(f, grad_f, alpha0, tol=1e-6, max_iter=1000):
    alpha = alpha0
    for _ in range(max_iter):
        if abs(grad_f(alpha)) < tol:
            break
        alpha = alpha - f(alpha) / grad_f(alpha)
    return alpha


print(newton_alpha(f, grad_f, 1.0))

def gradient_descent(x_0, f, df, max_iter=1000, tol=1e-6):
    x = x_0
    for i in range(max_iter): 
        fx = f(x)
        dfx = df(x)
        alpha = newton_alpha(f, grad_f, x, tol)
        dx = - alpha * dfx
        if np.linalg.norm(dx) < tol:
            break
        x += dx
    return x
"""

"""
# plan
bernstein(i,t)
dt_bernstein(i,t)
curve(P,t) = sum_i P[i] * bernstein(i,t)
dt_curve(P,t) = sum_i P[i] * dt_bernstein(i,t)
f(P,T) = 0.5 * sum_k np.norm(curve(P,T[k]) - X[k])**2
dt_f(P,T)[k] = np.dot(curve(P,T[k]) - X[k], dt_curve(P,T[k]))
dP_f(P,T)[i] = sum_k (curve(P,T[k]) - X[k]) * bernstein(i,T[k])
da_phi(a) = ...
daa_phi(a) = ...
"""

# variables définis dans le main
degree = 2
nodes = np.array([0,0,0,1,1,1])
X = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])

def bspline_basis(i, degree, t):
    n = len(nodes) - 1
    if degree == 0:
        if i >= n:
            return 0.0
        return 1.0 if nodes[i] <= t < nodes[i + 1] or (t == nodes[-1] and nodes[i + 1] == nodes[-1]) else 0.0
    first_part = 0.0
    second_part = 0.0
    if (i + degree) < n:
        denom1 = nodes[i + degree] - nodes[i]
        if denom1 != 0:
            first_part = (t - nodes[i]) / denom1 * bspline_basis(i, degree - 1, t)
    if (i + degree + 1) < n:
        denom2 = nodes[i + degree + 1] - nodes[i + 1]
        if denom2 != 0:
            second_part = (
                (nodes[i + degree + 1] - t)
                / denom2
                * bspline_basis(i + 1, degree - 1, t)
            )
    return first_part + second_part
 
def dt_bspline_basis(i, degree, t):
    first_part, second_part = 0.0, 0.0
    denom1 = nodes[i + degree] - nodes[i]
    if denom1 != 0:
        first_part  = degree / denom1 * bspline_basis(i, degree - 1, t)
    denom2 = nodes[i + degree + 1] - nodes[i + 1]
    if denom2 != 0:
        second_part = degree / denom2 * bspline_basis(i + 1, degree - 1, t)
    return first_part - second_part
 
def curve(P, t):
    t_min = nodes[degree]
    t_max = nodes[-degree - 1]
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = bspline_basis(i, degree, t)
        curve += N * P[i]
    return curve
 
def dt_curve(P, t):
    t_min = nodes[degree]
    t_max = nodes[-degree - 1]
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = dt_bspline_basis(i, degree, t)
        curve += N * P[i]
    return curve

def f(P, T):
    tot = 0.0
    for k in range(len(T)):
        tot += np.linalg.norm((curve(P, T[k]) - X[k])) ** 2
    return 0.5 * tot
 
def dt_f(P, T):
    n = len(T)
    grad_t = np.zeros(n)
    for k in range(n):   
        grad_t[k] = np.dot(curve(P, T[k]) - X[k], dt_curve(P, T[k]))
    return grad_t

def dP_f(P, T):
    m = P.shape[0]
    n = len(T)
    grad_P = np.zeros_like(P)

    for k in range(n):
        d = curve(P, T[k]) - X[k]
        for i in range(m):
            B_ik = bspline_basis(i, degree, T[k])
            grad_P[i] += d * B_ik
    return grad_P

# alpha dans R
# phi(alpha) = f(P - alpha * dP_f(P, T, X, t_knots, k), T)

def d_phi(alpha, P, T):
    D    = -dP_f(P, T)
    dphi = 0.0
    for k in range(len(T)):
        r_k = curve(P + alpha * D, T[k]) - X[k]
        d_k = curve(D, T[k])
        dphi += np.dot(r_k, d_k)
    return dphi

def dd_phi(alpha, P, T):
    D     = -dP_f(P, T)
    ddphi = 0.0
    for k in range(len(T)):
        d_k = curve(D, T[k])
        ddphi += np.dot(d_k, d_k)
    return ddphi