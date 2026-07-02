import numpy as np
from .curves import *

class Optimizers:

    ##
    # function: __init__
    #
    # description:
    #   Initializes Optimizers with a given curve.
    #
    # input:
    #   - curve : a Bézier or a Bspline curve
    #
    # output:
    #   - none
    ##
    def __init__(self, curve: Curve):
        self.curve = curve
        if self.curve.degree < 0:
            raise ValueError("Curve cannot have negative degree")

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
    #
    # output:
    #   - squared point distance error
    ##
    def err_PD(self, Xk, t):
        P_t = self.curve.eval(t)
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
    #
    # output:
    #   - first derivative of the error
    ##
    def err_PD_prim(self, Xk, t):
        P_t = self.curve.eval(t)
        P_t_prime = self.curve.derivative(t)
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
    #
    # output:
    #   - second derivative of the error
    ##
    def err_PD_primprim(self, Xk, t):
        P_t = self.curve.eval(t)
        P_t_prime = self.curve.derivative(t)
        P_t_second = self.curve.second_derivative(t)
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
    #   - initial_guess : starting value for Newton iteration
    #   - tol : convergence tolerance
    #   - max_iter : maximum number of iterations
    #
    # output:
    #   - optimized parameter tk
    ##
    def newton_tk(self, err_prim, err_primprim, Xk, initial_guess,tol=1e-6, max_iter=100):
        if isinstance(self.curve, BezierCurve):
            t_min = 0
            t_max = 1
        elif isinstance(self.curve, BSplineCurve):
            t_min = self.curve.knots[self.curve.degree]
            t_max = self.curve.knots[-(self.curve.degree + 1)]
        t = np.clip(initial_guess, t_min + 1e-8, t_max - 1e-8)
        
        for _ in range(max_iter):
            Fp  = err_prim(Xk, t)
            Fpp = err_primprim(Xk, t)
            if abs(Fpp) < 1e-14:
                break
            if isinstance(self.curve, BezierCurve):
                t_new = np.clip(t - Fp / Fpp, 0.0, 1.0)
            elif isinstance(self.curve, BSplineCurve):
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
    #   - initial_guesses : optional initial parameter values
    #   - n_samples : number of samples for initialization
    #
    # output:
    #   - list of optimized foot points tk for every Xk
    ##
    def all_tk(self, X, initial_guesses=None, n_samples=50):
        if isinstance(self.curve, BezierCurve):
            t_min = 0
            t_max = 1
        elif isinstance(self.curve, BSplineCurve):
            t_min = self.curve.knots[self.curve.degree]
            t_max = self.curve.knots[-(self.curve.degree + 1)]
        # First iteration
        # initial guesses are estimated automatically
        if initial_guesses is None: 
            t_samples = np.linspace(t_min, t_max, n_samples, endpoint=False)
            initial_guesses = []
            for Xk in X:
                dists = [self.err_PD(Xk, t) for t in t_samples]
                initial_guesses.append(t_samples[np.argmin(dists)])
        # Next iterations
        # previous tk values are reused
        return [self.newton_tk(self.err_PD_prim, self.err_PD_primprim, Xk, t0) for Xk, t0 in zip(X, initial_guesses)]

    ##
    # function: newton_alpha
    #
    # description:
    #   Uses Newton's method to compute the gradient descent step.
    #
    # input:
    #   - d_phi_func : first derivative of line-search function φ
    #   - dd_phi_func : second derivative of φ
    #   - T : parameter values
    #   - X : data points
    #   - alpha0 : initial step size
    #   - tol : convergence tolerance
    #   - max_iter : maximum number of iterations
    #
    # output:
    #   - optimized alpha value
    ##
    def newton_alpha(self, d_phi_func, dd_phi_func, X, T, alpha0, tol=1e-6, max_iter=100):
        alpha = alpha0
        for _ in range(max_iter):
            grad = d_phi_func(alpha, X, T)
            if abs(grad) < tol:
                break
            denom = dd_phi_func(alpha, X, T)
            if denom == 0:
                break
            alpha = alpha - grad / denom
        return alpha
        
    ##
    # function: avg_error
    #
    # description:
    #   Computes the average error.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - average fitting error
    ##
    def avg_error(self, X, T):
        return np.sqrt(sum(self.err_PD(X[k], T[k]) for k in range(len(T)))/len(X))

    ##
    # function: max_error
    #
    # description:
    #   Computes the max error.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - average fitting error
    ##
    def max_error(self, X, T):
        max_err = 0.0
        for tk, Xk in zip(T, X):
            d = np.linalg.norm(self.curve.eval(tk) - Xk)
            if d > max_err:
                max_err = d
        return max_err
    
class PDM(Optimizers):

    ##
    # function: f_PD
    #
    # description:
    #   Computes the Point Distance Minimization (PDM)
    #   objective function.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - PDM's objective function value
    ##
    def f_PD(self, X, T):
        return 0.5 * sum(self.err_PD(X[k], T[k]) for k in range(len(T)))
    
    ##
    # function: dP_f_PD
    #
    # description:
    #   Computes the gradient of the PDM objective
    #   function with respect to the control points.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - gradient matrix
    ##
    def dP_f_PD(self, X, T):
        m = self.curve.P.shape[0]
        n = len(T)
        grad_P = np.zeros_like(self.curve.P)

        for k in range(n):
            d = self.curve.eval(T[k]) - X[k]
            for i in range(m):
                B_ik = self.curve.basis(i, self.curve.degree, T[k])
                grad_P[i] += d * B_ik
        return grad_P
    
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
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - PDM's objective function value
    ##
    def phi_PD(self, alpha, X, T):
        D = -self.dP_f_PD(X, T)
        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(self.curve.P + alpha * D)
        elif  isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
        temp_PDM = PDM(temp_curve)
        return temp_PDM.f_PD(X, T)
    
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
    def d_phi_PD(self, alpha, X, T):
        D = -self.dP_f_PD(X, T)
        dphi = 0.0
        if isinstance(self.curve, BezierCurve):
            temp_curve1 = BezierCurve(self.curve.P + alpha * D)
            temp_curve2 = BezierCurve(D)
        elif  isinstance(self.curve, BSplineCurve):
            temp_curve1 = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
            temp_curve2 = BSplineCurve(D, self.curve.knots, self.curve.degree)
        for tk, Xk in zip(T, X):
            r = temp_curve1.eval(tk) - Xk
            d = temp_curve2.eval(tk)
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
    def dd_phi_PD(self, alpha, X, T):
        D = -self.dP_f_PD(X, T)
        ddphi = 0.0
        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(D)
        elif  isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(D, self.curve.knots, self.curve.degree)
        for tk in T:
            d = temp_curve.eval(tk)
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
    #   - logarithm of maximum fitting errors
    ##
    def gradient_descent_PD(self, X, T0, alpha0=0.1,max_iter=100,tol=1e-6,constraint="opened"):
        P = self.curve.P.copy()
        if isinstance(self.curve, BezierCurve):
            opt_curve = BezierCurve(P)
        elif isinstance(self.curve, BSplineCurve):
            opt_curve = BSplineCurve(P, self.curve.knots, self.curve.degree)
        opt_pdm = PDM(opt_curve)

        T = np.asarray(T0, dtype=float).copy()
        log_avg_error,  log_max_error, log_iter= [], [], []

        for i in range(max_iter):
            max_err = np.log10(opt_pdm.max_error(X, T))
            log_iter.append(np.log10(i + 1))
            log_avg_error.append(np.log10(opt_pdm.avg_error(X, T)))
            log_max_error.append(max_err)

            grad_P = opt_pdm.dP_f_PD(X, T)
            norm_grad = np.linalg.norm(grad_P)

            # Convergence test
            if norm_grad < tol or max_err < np.log10(0.03):
                print(f"Convergence achieved after {i+1} iterations")
                break

            # Descent direction
            D = -grad_P

            # Newton line-search
            alpha = opt_pdm.newton_alpha(opt_pdm.d_phi_PD,opt_pdm.dd_phi_PD,X,T,alpha0,tol)

            # Fallback step size
            if alpha <= 0 or np.isnan(alpha):
                alpha = alpha0
            
            # Update control points
            opt_pdm.curve.P += alpha * D
            # constraints can be modified in need
            if constraint=="opened":
                opt_pdm.curve.P[0] = self.curve.P[0]
                opt_pdm.curve.P[-1] = self.curve.P[-1]
            elif constraint=="closed":
                opt_pdm.curve.P[-1] = opt_pdm.curve.P[0].copy()
            
            # Recompute footpoint parameters
            T = opt_pdm.all_tk(X, initial_guesses=T)
        return opt_pdm.curve.P, log_iter, log_avg_error, log_max_error
    
class TDM(Optimizers):

    ##
    # function: err_TD
    #
    # description:
    #   Computes the Tangent Distance (TD) error
    #   between a curve point P(t) and a data
    #   point Xk.
    #
    # input:
    #   - Xk : data point
    #   - t : parameter value
    #
    # output:
    #   - tangent distance error
    ##
    def err_TD(self, Xk, t):
        P_t = self.curve.eval(t)
        n = self.curve.unit_normal(t)
        return np.dot(P_t - Xk, n) ** 2  

    ##
    # function: f_TD
    #
    # description:
    #   Computes the global Tangent Distance Minimization
    #   (TDM) objective function.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - TDM objective function value
    ##
    def f_TD(self, X, T):
        return 0.5 * sum(self.err_TD(X[k], T[k]) for k in range(len(T)))

    ##
    # function: dP_f_TD
    #
    # description:
    #   Computes the gradient of the TDM objective
    #   function with respect to the control points.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - gradient matrix
    ##
    def dP_f_TD(self, X, T):
        m = self.curve.P.shape[0]
        grad_P = np.zeros_like(self.curve.P)

        for k in range(len(T)):
            diff = self.curve.eval(T[k]) - X[k]
            n_k = self.curve.unit_normal(T[k])
            d_proj = np.dot(n_k, diff)
            for i in range(m):
                B_ik = self.curve.basis(i, self.curve.degree, T[k])
                grad_P[i] += d_proj * B_ik * n_k
        return grad_P

    ##
    # function: phi_TD
    #
    # description:
    #   Evaluates the line-search function phi
    #   used in the Tangent Distance Minimization (TDM).
    #
    # input:
    #   - alpha : step size
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - phi value
    ##
    def phi_TD(self, alpha, X, T):
        D = -self.dP_f_TD(X, T)
        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(self.curve.P + alpha * D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
        temp_tdm = TDM(temp_curve)
        return temp_tdm.f_TD(X, T)

    ##
    # function: d_phi_TD
    #
    # description:
    #   Computes the first derivative of the
    #   line-search function phi for TDM.
    #
    # input:
    #   - alpha : step size
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - first derivative of phi
    ##
    def d_phi_TD(self, alpha, X, T):
        D = -self.dP_f_TD(X, T)
        dphi = 0.0
        if isinstance(self.curve, BezierCurve):
            temp_curve1 = BezierCurve(self.curve.P + alpha * D)
            temp_curve2 = BezierCurve(D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve1 = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
            temp_curve2 = BSplineCurve(D, self.curve.knots, self.curve.degree)
        for tk, Xk in zip(T, X):
            n_k = self.curve.unit_normal(tk)
            r_k = temp_curve1.eval(tk) - Xk
            d_k = np.dot(n_k, temp_curve2.eval(tk))
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
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - second derivative of phi
    ##
    def dd_phi_TD(self, alpha, X, T):
        D = -self.dP_f_TD(X, T)
        ddphi = 0.0
        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(D, self.curve.knots, self.curve.degree)
        for tk in T:
            n_k = self.curve.unit_normal(tk)
            d_k = np.dot(n_k, temp_curve.eval(tk))
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
    #   - X : data points
    #   - T0 : initial parameter values
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
    #   - logarithm of maximum fitting errors
    ##
    def gradient_descent_TD(self, X, T0, alpha0=0.1, max_iter=100, tol=1e-6, constraint="opened"):
        P = self.curve.P.copy()
        if isinstance(self.curve, BezierCurve):
            opt_curve = BezierCurve(P)
        elif isinstance(self.curve, BSplineCurve):
            opt_curve = BSplineCurve(P, self.curve.knots, self.curve.degree)
        opt_tdm = self.__class__(opt_curve)

        T = np.asarray(T0, dtype=float).copy()
        log_avg_error, log_max_error, log_iter = [], [], []

        for i in range(max_iter):
            max_err = np.log10(opt_tdm.max_error(X, T))
            log_iter.append(np.log10(i + 1))
            log_avg_error.append(np.log10(opt_tdm.avg_error(X, T)))
            log_max_error.append(max_err)

            grad_P = opt_tdm.dP_f_TD(X, T)
            norm_grad = np.linalg.norm(grad_P)

            # Convergence test
            if norm_grad < tol or max_err < np.log10(0.03):
                print(f"Convergence achieved after {i+1} iterations")
                break

            # Descent direction
            D = -grad_P

            # Newton line-search
            alpha = opt_tdm.newton_alpha(opt_tdm.d_phi_TD, opt_tdm.dd_phi_TD, X, T, alpha0, tol)

            # Fallback step size
            if alpha <= 0 or np.isnan(alpha):
                alpha = alpha0

            # Update control points
            opt_tdm.curve.P += alpha * D
            # constraints can be modified in need
            if constraint == "opened":
                opt_tdm.curve.P[0] = self.curve.P[0]
                opt_tdm.curve.P[-1] = self.curve.P[-1]
            elif constraint == "closed":
                opt_tdm.curve.P[-1] = opt_tdm.curve.P[0].copy()

            # Recompute footpoint parameters
            T = opt_tdm.all_tk(X, initial_guesses=T)
        return opt_tdm.curve.P, log_iter, log_avg_error, log_max_error

class SDM(Optimizers):

    ##
    # function: signed_distance
    #
    # description:
    #   Computes the signed distance between
    #   a curve point P(t) and a data point Xk.
    #
    #   The sign is determined using the curve
    #   normal direction:
    #   - positive on one side of the curve
    #   - negative on the other side
    #
    # input:
    #   - Xk : data point
    #   - t : parameter value
    #
    # output:
    #   - signed distance value
    ##
    def signed_distance(self, Xk, t):
        P_t = self.curve.eval(t)
        d = np.linalg.norm(P_t - Xk)
        n = self.curve.unit_normal(t)
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
    #   - if the signed distance is negative,
    #     the full SDM formulation is used
    #
    #   - otherwise, the method falls back to the
    #     tangent distance formulation
    #
    # input:
    #   - Xk : data point
    #   - t : parameter value
    #
    # output:
    #   - SDM local error value
    ##
    def err_SD(self, Xk, t):
        d = self.signed_distance(Xk, t)
        if d < 0:
            rho = self.curve.curvature_radius(t)
            unit_T = self.curve.unit_tangent(t)
            unit_N = self.curve.unit_normal(t)
            P_t = self.curve.eval(t)
            coeff = d / (d - rho)
            diff = P_t - Xk

            first_part = coeff * np.dot(diff, unit_T) ** 2
            second_part = np.dot(diff, unit_N) ** 2
            return first_part + second_part
        else:
            # return error_TD
            P_t = self.curve.eval(t)
            unit_N = self.curve.unit_normal(t)
            return np.dot(P_t - Xk, unit_N) ** 2

    ##
    # function: f_SD
    #
    # description:
    #   Computes the global Squared Distance Minimization
    #   (SDM) objective function.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - SDM value
    ##
    def f_SD(self, X, T):
        return 0.5 * sum(self.err_SD(X[k], T[k]) for k in range(len(T)))

    ##
    # function: dP_f_SD
    #
    # description:
    #   Computes the gradient of the SDM objective
    #   function with respect to the control points.
    #
    # input:
    #   - X : data points
    #   - T : parameter values
    #
    # output:
    #   - gradient matrix
    ##
    def dP_f_SD(self, X, T):
        m = self.curve.P.shape[0]
        grad_P = np.zeros_like(self.curve.P)

        for k in range(len(T)):
            tk = T[k]
            Xk = X[k]
            signed_d = self.signed_distance(Xk, tk)
            diff = self.curve.eval(tk) - Xk
            unit_N = self.curve.unit_normal(tk)
            d_projN = np.dot(unit_N, diff)

            if signed_d < 0:
                rho = self.curve.curvature_radius(tk)
                unit_T = self.curve.unit_tangent(tk)
                coeff = signed_d / (signed_d - rho)
                d_projT = np.dot(unit_T, diff)
                for i in range(m):
                    B_ik = self.curve.basis(i, self.curve.degree, tk)
                    grad_P[i] += coeff * d_projT * B_ik * unit_T

            for i in range(m):
                if isinstance(self.curve, BezierCurve):
                    B_ik = self.curve.basis(i, self.curve.degree, tk)
                elif isinstance(self.curve, BSplineCurve):
                    B_ik = self.curve.basis(i, self.curve.degree, tk)
                grad_P[i] += d_projN * B_ik * unit_N

        return grad_P

    ##
    # function: phi_SD
    #
    # description:
    #   Evaluates the line-search function phi
    #   used in the Squared Distance Minimization (SDM).
    #   This function is not explicitly used here, but
    #   its derivatives are required for the
    #   Newton line-search procedure.
    ##
    def phi_SD(self, alpha, X, T):
        D = -self.dP_f_SD(X, T)
        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(self.curve.P + alpha * D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
        temp_sdm = SDM(temp_curve)
        return temp_sdm.f_SD(X, T)

    ##
    # function: d_phi_SD
    #
    # description:
    #   Computes the first derivative of the
    #   line-search function phi for SDM.
    ##
    def d_phi_SD(self, alpha, X, T):
        D = -self.dP_f_SD(X, T)
        dphi = 0.0

        if isinstance(self.curve, BezierCurve):
            temp_curve1 = BezierCurve(self.curve.P + alpha * D)
            temp_curve2 = BezierCurve(D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve1 = BSplineCurve(self.curve.P + alpha * D, self.curve.knots, self.curve.degree)
            temp_curve2 = BSplineCurve(D, self.curve.knots, self.curve.degree)

        for k in range(len(T)):
            tk = T[k]
            Xk = X[k]
            signed_d = self.signed_distance(Xk, tk)
            diff = temp_curve1.eval(tk) - Xk

            if signed_d < 0:
                rho = self.curve.curvature_radius(tk)
                unit_T = self.curve.unit_tangent(tk)
                coeff = signed_d / (signed_d - rho)
                r1 = coeff * np.dot(unit_T, diff)
                d1 = np.dot(unit_T, temp_curve2.eval(tk))
                dphi += r1 * d1

            unit_N = self.curve.unit_normal(tk)
            r = np.dot(unit_N, diff)
            d = np.dot(unit_N, temp_curve2.eval(tk))
            dphi += r * d

        return dphi

    ##
    # function: dd_phi_SD
    #
    # description:
    #   Computes the second derivative of the
    #   line-search function phi for SDM.
    ##
    def dd_phi_SD(self, alpha, X, T):
        D = -self.dP_f_SD(X, T)
        ddphi = 0.0

        if isinstance(self.curve, BezierCurve):
            temp_curve = BezierCurve(D)
        elif isinstance(self.curve, BSplineCurve):
            temp_curve = BSplineCurve(D, self.curve.knots, self.curve.degree)

        for k in range(len(T)):
            tk = T[k]
            Xk = X[k]
            signed_d = self.signed_distance(Xk, tk)

            if signed_d < 0:
                rho = self.curve.curvature_radius(tk)
                unit_T = self.curve.unit_tangent(tk)
                coeff = signed_d / (signed_d - rho)
                d = np.dot(unit_T, temp_curve.eval(tk))
                ddphi += coeff * (d ** 2)

            unit_N = self.curve.unit_normal(tk)
            d_k = np.dot(unit_N, temp_curve.eval(tk))
            ddphi += d_k ** 2

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
    #   - X : data points
    #   - T0 : initial parameter values
    #   - alpha0 : initial step size
    #   - max_iter : maximum iterations
    #   - tol : convergence tolerance
    #   - constraint : for open or closed curves
    #
    # output:
    #   - optimized control points
    #   - logarithm of iteration indices
    #   - logarithm of average fitting errors
    #   - logarithm of maximum fitting errors
    ##
    def gradient_descent_SD(self, X, T0, alpha0=0.1, max_iter=100, tol=1e-6, constraint="opened"):
        P = self.curve.P.copy()
        if isinstance(self.curve, BezierCurve):
            opt_curve = BezierCurve(P)
        elif isinstance(self.curve, BSplineCurve):
            opt_curve = BSplineCurve(P, self.curve.knots, self.curve.degree)
        opt_sdm = SDM(opt_curve)

        T = np.asarray(T0, dtype=float).copy()
        log_avg_error, log_max_error, log_iter = [], [], []

        for i in range(max_iter):
            max_err = np.log10(opt_sdm.max_error(X, T))
            log_iter.append(np.log10(i + 1))
            log_avg_error.append(np.log10(opt_sdm.avg_error(X, T)))
            log_max_error.append(max_err)

            grad_P = opt_sdm.dP_f_SD(X, T)
            norm_grad = np.linalg.norm(grad_P)

            # Convergence test
            if norm_grad < tol or max_err < np.log10(0.03):
                print(f"Convergence achieved after {i+1} iterations")
                break

            # Descent direction
            D = -grad_P

            # Newton line-search
            alpha = opt_sdm.newton_alpha(opt_sdm.d_phi_SD, opt_sdm.dd_phi_SD, X, T, alpha0, tol)

            # Fallback step size
            if alpha <= 0 or np.isnan(alpha):
                alpha = alpha0

            # Update control points
            opt_sdm.curve.P += alpha * D
            # constraints can be modified in need
            if constraint == "opened":
                opt_sdm.curve.P[0] = self.curve.P[0]
                opt_sdm.curve.P[-1] = self.curve.P[-1]
            elif constraint == "closed":
                opt_sdm.curve.P[-1] = opt_sdm.curve.P[0].copy()

            # Recompute footpoint parameters
            T = opt_sdm.all_tk(X, initial_guesses=T)

        return opt_sdm.curve.P, log_iter, log_avg_error, log_max_error