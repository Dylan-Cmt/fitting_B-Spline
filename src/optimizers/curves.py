import numpy as np
from math import comb

##
# class: Curve
#
# description:
#   Base class for parametric curves (e.g. Bézier, B-spline, NURBS).
#   Defines the common interface that all curve types must implement.
##

class Curve:

    ##
    # function: basis
    #
    # description:
    #   Computes a basis for a Bézier or a BSpline curve
    #
    # input:
    #   - i : basis index
    #   - n : polynomial degree
    #   - t : parameter in [0,1]
    #
    # output:
    #   - Basis for Bézier or BSpline curve
    ##
    def basis(self, i, n, t):
        raise NotImplementedError()

    ##
    # function: eval
    #
    # description:
    #   Evaluates the curve at parameter t.
    #
    # input:
    #   - t : parameter in [0, 1]
    #
    # output:
    #   - point on the curve
    ##
    def eval(self, t):
        raise NotImplementedError()

    ##
    # function: derivative
    #
    # description:
    #   Computes the first derivative of the curve.
    #
    # input:
    #   - t : parameter in [0, 1]
    #
    # output:
    #   - tangent vector (not normalized)
    ##
    def derivative(self, t):
        raise NotImplementedError()

    ##
    # function: second_derivative
    #
    # description:
    #   Computes the second derivative of the curve.
    #
    # input:
    #   - t : parameter in [0, 1]
    #
    # output:
    #   - second derivative vector
    ##
    def second_derivative(self, t):
        raise NotImplementedError()

    ##
    # function: degree_elevate
    #
    # description:
    #   Performs degree elevation of the curve.
    #   Implementation depends on curve type.
    #
    # input:
    #   - none
    #
    # output:
    #   - elevated curve representation
    ##
    def degree_elevate(self):
        raise NotImplementedError()

    ##
    # function: degree_elevate_multiple
    #
    # description:
    #   Performs degree elevation of the curve by k.
    #   Implementation depends on curve type.
    #
    # input:
    #   - k : number of degree elevations
    #
    # output:
    #   - elevated curve representation
    ## 
    def degree_elevate_multiple(self, k):
        raise NotImplementedError()

    ##
    # function: unit_tangent
    #
    # description:
    #   Computes the unit tangent vector of the curve.
    #
    # input:
    #   - t : parameter in [0, 1]
    #
    # output:
    #   - unit tangent vector
    ##
    def unit_tangent(self, t):
        if self.degree < 1:
            raise ValueError("Invalid Bézier degree: tangent computation requires degree >= 1.")
        T = self.derivative(t)
        eps = 1e-12
        return np.asarray(T / (np.linalg.norm(T) + eps), dtype=float)


    ##
    # function: unit_normal
    #
    # description:
    #   Computes the unit normal vector (2D).
    #   Obtained by rotating the tangent by 90°.
    #
    # input:
    #   - t : parameter in [0, 1]
    #
    # output:
    #   - unit normal vector
    ##
    def unit_normal(self, t):
        uT = self.unit_tangent(t)
        return np.asarray([uT[1], -uT[0]], dtype=float)

    ##
    # function: curvature2D
    #
    # description:
    #   Computes the signed curvature of a 2D parametric curve.
    #   Uses first and second derivatives of the curve.
    #
    # input:
    #   - t : parameter in [0,1]
    #
    # output:
    #   - curvature value κ(t)
    ##
    def curvature2D(self, t):
        P_t_prime = self.derivative(t)
        P_t_second = self.second_derivative(t)
        denom = np.linalg.norm(P_t_prime) ** 3
        if denom < 1e-12:
            return 1e12
        num = np.linalg.det(np.vstack((P_t_prime, P_t_second)))
        return abs(num) / denom

    ##
    # function: curvature_radius
    #
    # description:
    #   Computes the radius of curvature of the curve.
    #   Defined as the inverse of curvature.
    #   Handles near-zero curvature for numerical stability.
    #
    # input:
    #   - t : parameter in [0,1]
    #
    # output:
    #   - radius of curvature R(t)
    ##
    def curvature_radius(self, t):
        K = self.curvature2D(t)
        if K < 1e-12:
            return 1e12
        elif K >= 1e12:
            return 0
        return 1.0 / K

##
# class: BezierCurve
#
# description:
#   Represents a Bézier curve defined by a set of control points.
#   Inherits from the base Curve class and implements all curve operations
#   using Bernstein polynomial formulation.
##
class BezierCurve(Curve):

    ##
    # function: __init__
    #
    # description:
    #   Initializes a Bézier curve with a given set of control points.
    #
    # input:
    #   - control_points : list or array of control points
    #
    # output:
    #   - none
    ##
    def __init__(self, control_points):
        self.P = np.asarray(control_points, dtype=float)
        if self.P.size == 0:
            raise ValueError("Control points cannot be empty")
        self.degree = len(self.P) - 1

    ##
    # function: basis
    #
    # description:
    #   Computes a Bernstein basis polynomial:
    #
    #   B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
    #
    # input:
    #   - i : Bernstein basis index
    #   - n : Bézier polynomial degree
    #   - t : parameter in [0,1]
    #
    # output:
    #   - value of the Bernstein basis B_i^n(t)
    #
    # errors:
    #   - ValueError if n < 0
    ##
    def basis(self, i, n, t):
        if n < 0:
            raise ValueError("Bezier degree must be non-negative")
        if i < 0 or i > n:
            return 0.0
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    ##
    # function: _bernstein_basis_vector
    #
    # description:
    #   Generates the complete vector of Bernstein
    #   basis functions:
    #
    #   [B_0^n(t), ..., B_n^n(t)]
    #
    # input:
    #   - n : Bézier polynomial degree
    #   - t : scalar or array of parameters
    #
    # output:
    #   - numpy array containing all Bernstein bases
    ##
    def _bernstein_basis_vector(self, n, t):
        t = np.asarray(t, dtype=float)
        return np.array([self.basis(i, n, t) for i in range(n + 1)])

    ##
    # function: eval
    #
    # description:
    #   Evaluates a Bézier curve from its control
    #   points:
    #
    #   P(t) = Σ B_i^n(t) * P_i
    #
    # input:
    #   - control_points : array of control points
    #   - t : scalar or array of parameters
    #
    # output:
    #   - curve point if t is scalar
    #   - array of curve points if t is a vector
    #
    # errors:
    #   - ValueError if control points are empty
    ##
    def eval(self, t):
        t = np.asarray(t, dtype=float)
        n = self.degree
        if t.ndim == 0:
            B = self._bernstein_basis_vector(n, t)
            return B @ self.P
        curve = np.zeros((len(t), self.P.shape[1]))
        for idx, tt in enumerate(t):
            B = self._bernstein_basis_vector(n, tt)
            curve[idx] = B @ self.P
        return curve

    ##
    # function: derivative
    #
    # description:
    #   Computes the first derivative of a Bézier curve.
    #
    #   The derivative corresponds to the hodograph:
    #
    #   P'(t) = n * Σ B_i^(n-1)(t) * (P_(i+1) - P_i)
    #
    # input:
    #   - control_points : array of control points
    #   - t : scalar or array parameter
    #
    # output:
    #   - derivative vector evaluated at t
    ##
    def derivative(self, t):
        n = self.degree
        if n <= 0:
            return np.zeros_like(self.P[0])
        D = n * (self.P[1:] - self.P[:-1])
        return BezierCurve(D).eval(t)

    ##
    # function: second_derivative
    #
    # description:
    #   Computes the second derivative of a Bézier curve.
    #
    # input:
    #   - control_points : array of control points
    #   - t : scalar or array parameter
    #
    # output:
    #   - second derivative evaluated at t
    ##
    def second_derivative(self, t):
        n = self.degree
        if n <= 1:
            return np.zeros_like(self.P[0])
        D1 = n * (self.P[1:] - self.P[:-1])
        D2 = (n - 1) * (D1[1:] - D1[:-1])
        return BezierCurve(D2).eval(t)
    
    ##
    # function: degree_elevate
    #
    # description:
    #   Elevates the degree of a Bézier curve by one.
    #
    # input:
    #   - control_points : array of control points
    #
    # output:
    #   - array of control points for the elevated curve
    ##
    def degree_elevate(self):
        P = np.asarray(self.P, dtype=float)
        n = self.degree
        if n < 0:
            raise ValueError("Control points cannot be empty")
        elif n == 0:
            return np.array([P[0], P[0]])
        # Degree elevation formula
        new_control_points = np.zeros((n + 2, P.shape[1]))
        new_control_points[0] = P[0]
        new_control_points[-1] = P[-1]
        for i in range(1, n + 1):
            coeff = i / (n + 1)
            new_control_points[i] = coeff * P[i - 1] + (1 - coeff) * P[i]
        return new_control_points
    
    ##
    # function: degree_elevate_multiple
    #
    # description:
    #   Elevates the degree of a Bézier curve by k.
    #
    # input:
    #   - control_points : array of control points
    #   - k : number of degree elevations
    #
    # output:
    #   - array of control points for the elevated curve
    ##
    def degree_elevate_multiple(self, k):
        P = np.asarray(self.P, dtype=float)
        for _ in range(k):
            P = BezierCurve(P).degree_elevate()
        return P

##
# class: BSplineCurve
#
# description:
#   Represents a B-spline curve defined by control points, a knot vector,
#   and a polynomial degree.
#   Inherits from the base Curve class.
##
class BSplineCurve(Curve):

    ##
    # function: __init__
    #
    # description:
    #   Initializes a B-spline curve with control points, knots, and degree.
    #
    # input:
    #   - control_points : array of control points
    #   - knots : knot vector
    #   - degree : polynomial degree of the B-spline
    #
    # output:
    #   - none
    ##
    def __init__(self, control_points, knots, degree):
        self.P = np.asarray(control_points, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.degree = int(degree)
        if self.P.size == 0:
            raise ValueError("Control points cannot be empty")

    ##
    # function: basis
    #
    # description:
    #   Computes the B-spline basis function N_{i,p}(t) recursively
    #   using the Cox–de Boor recursion formula.
    #
    # input:
    #   - i : basis function index
    #   - p : degree of the basis function
    #   - t : parameter in [0,1]
    #
    # output:
    #   - basis function value N_{i,p}(t)
    ##
    def basis(self, i, p, t):
        knots = self.knots
        n = len(knots) - 1
        if p == 0:
            if i >= n:
                return 0.0
            return 1.0 if (knots[i] <= t < knots[i + 1]) or (t == knots[-1] and knots[i + 1] == knots[-1]) else 0.0
        first, second = 0.0, 0.0
        if (i + p) < n:
            denom1 = knots[i + p] - knots[i]
            if denom1 != 0:
                first = (t - knots[i]) / denom1 * self.basis(i, p - 1, t)
        if (i + p + 1) < n:
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom2 != 0:
                second = (knots[i + p + 1] - t) / denom2 * self.basis(i + 1, p - 1, t)
        return first + second

    ##
    # function: _basis_dt
    #
    # description:
    #   Computes the first derivative of the B-spline basis function.
    #
    # input:
    #   - i : basis function index
    #   - p : degree
    #   - t : parameter in [0,1]
    #
    # output:
    #   - first derivative of N_{i,p}(t)
    ##
    def _basis_dt(self, i, p, t):
        if p <= 0:
            return 0.0
        knots = self.knots
        first, second = 0.0, 0.0
        denom1 = knots[i + p] - knots[i]
        if denom1 != 0:
            first = p / denom1 * self.basis(i, p - 1, t)
        denom2 = knots[i + p + 1] - knots[i + 1]
        if denom2 != 0:
            second = p / denom2 * self.basis(i + 1, p - 1, t)
        return first - second
    
    ##
    # function: _basis_dtt
    #
    # description:
    #   Computes the second derivative of the B-spline basis function.
    #
    # input:
    #   - i : basis function index
    #   - p : degree
    #   - t : parameter in [0,1]
    #
    # output:
    #   - first derivative of N_{i,p}(t)
    ##
    def _basis_dtt(self, i, p, t):
        if p <= 1:
            return 0.0
        knots = self.knots
        first, second, third = 0.0, 0.0, 0.0
        denom1 = (knots[i + p] - knots[i]) * (knots[i + p - 1] - knots[i])
        if denom1 != 0:
            first = p * (p - 1) / denom1 * self.basis(i, p - 2, t)
        denom2a = (knots[i + p] - knots[i]) * (knots[i + p] - knots[i + 1])
        denom2b = (knots[i + p + 1] - knots[i + 1]) * (knots[i + p] - knots[i + 1])
        coeff2 = 0.0
        if denom2a != 0:
            coeff2 += 1.0 / denom2a
        if denom2b != 0:
            coeff2 += 1.0 / denom2b
        second = p * (p - 1) * coeff2 * self.basis(i + 1, p - 2, t)
        denom3 = (knots[i + p + 1] - knots[i + 1]) * (knots[i + p + 1] - knots[i + 2])
        if denom3 != 0:
            third = p * (p - 1) / denom3 * self.basis(i + 2, p - 2, t)
        return first - second + third

    ##
    # function: eval
    #
    # description:
    #   Evaluates the B-spline curve at parameter t.
    #   Computes the weighted sum of control points and basis functions.
    #
    # input:
    #   - t : parameter in [0,1]
    #
    # output:
    #   - point on the curve
    ##
    def eval(self, t):
        curve = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self.basis(i, self.degree, t)
            curve += N * self.P[i]
        return curve

    ##
    # function: derivative
    #
    # description:
    #   Computes the first derivative of the B-spline curve
    #   using derivatives of basis functions.
    #
    # input:
    #   - t : parameter in [0,1]
    #
    # output:
    #   - tangent vector
    ##
    def derivative(self, t):
        vec = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self._basis_dt(i, self.degree, t)
            vec += N * self.P[i]
        return vec

    ##
    # function: second_derivative
    #
    # description:
    #   Computes the second derivative of the B-spline curve
    #   using second derivatives of basis functions.
    #
    # input:
    #   - t : parameter in [0,1]
    #
    # output:
    #   - second derivative vector
    ##
    def second_derivative(self, t):
        vec = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self._basis_dtt(i, self.degree, t)
            vec += N * self.P[i]
        return vec

    ##
    # function: degree_elevate
    #
    # description:
    #   Elevates the degree of a B-spline curve by one.
    #
    # input:
    #   - control_points : array of control points
    #
    # output:
    #   - array of control points for the elevated curve
    ##
    def degree_elevate(self):
        raise NotImplementedError()
    
    ##
    # function: degree_elevate_multiple
    #
    # description:
    #   Elevates the degree of a B-spline curve by k.
    #
    # input:
    #   - control_points : array of control points
    #   - k : number of degree elevations
    #
    # output:
    #   - array of control points for the elevated curve
    ##
    def degree_elevate_multiple(self, k):
        raise NotImplementedError()