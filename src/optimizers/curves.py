import numpy as np
from math import comb


class Curve:
    def eval(self, t):
        raise NotImplementedError()

    def derivative(self, t):
        raise NotImplementedError()

    def second_derivative(self, t):
        raise NotImplementedError()

    def unit_tangent(self, t):
        if self.degree < 1:
            raise ValueError("Invalid Bézier degree: tangent computation requires degree >= 1.")
        T = self.derivative(t)
        eps = 1e-12
        return np.asarray(T / (np.linalg.norm(T) + eps), dtype=float)

    def unit_normal(self, t):
        uT = self.unit_tangent(t)
        return np.asarray([uT[1], -uT[0]], dtype=float)

    def curvature2D(self, t):
        P_t_prime = self.derivative(t)
        P_t_second = self.second_derivative(t)
        denom = np.linalg.norm(P_t_prime) ** 3
        if denom < 1e-12:
            return 1e12
        num = np.linalg.det(np.vstack((P_t_prime, P_t_second)))
        return abs(num) / denom

    def curvature_radius(self, t):
        K = self.curvature2D(t)
        if K < 1e-12:
            return 1e12
        elif K >= 1e12:
            return 0
        return 1.0 / K


class BezierCurve(Curve):
    def __init__(self, control_points):
        self.P = np.asarray(control_points, dtype=float)
        if self.P.size == 0:
            raise ValueError("Control points cannot be empty")
        self.degree = len(self.P) - 1

    def _bernstein(self, i, n, t):
        if n < 0:
            raise ValueError("Bezier degree must be non-negative")
        if i < 0 or i > n:
            return 0.0
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def _bernstein_vector(self, n, t):
        t = np.asarray(t, dtype=float)
        return np.array([self._bernstein(i, n, t) for i in range(n + 1)])

    def eval(self, t):
        t = np.asarray(t, dtype=float)
        n = self.degree
        if t.ndim == 0:
            B = self._bernstein_vector(n, t)
            return B @ self.P
        curve = np.zeros((len(t), self.P.shape[1]))
        for idx, tt in enumerate(t):
            B = self._bernstein_vector(n, tt)
            curve[idx] = B @ self.P
        return curve

    def derivative(self, t):
        n = self.degree
        if n <= 0:
            return np.zeros_like(self.P[0])
        D = n * (self.P[1:] - self.P[:-1])
        return BezierCurve(D).eval(t)

    def second_derivative(self, t):
        n = self.degree
        if n <= 1:
            return np.zeros_like(self.P[0])
        D1 = n * (self.P[1:] - self.P[:-1])
        D2 = (n - 1) * (D1[1:] - D1[:-1])
        return BezierCurve(D2).eval(t)


class BSplineCurve(Curve):
    def __init__(self, control_points, knots, degree):
        self.P = np.asarray(control_points, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.degree = int(degree)
        if self.P.size == 0:
            raise ValueError("Control points cannot be empty")

    def _basis(self, i, p, t):
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
                first = (t - knots[i]) / denom1 * self._basis(i, p - 1, t)
        if (i + p + 1) < n:
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom2 != 0:
                second = (knots[i + p + 1] - t) / denom2 * self._basis(i + 1, p - 1, t)
        return first + second

    def _basis_dt(self, i, p, t):
        if p <= 0:
            return 0.0
        knots = self.knots
        first, second = 0.0, 0.0
        denom1 = knots[i + p] - knots[i]
        if denom1 != 0:
            first = p / denom1 * self._basis(i, p - 1, t)
        denom2 = knots[i + p + 1] - knots[i + 1]
        if denom2 != 0:
            second = p / denom2 * self._basis(i + 1, p - 1, t)
        return first - second

    def _basis_dtt(self, i, p, t):
        if p <= 1:
            return 0.0
        knots = self.knots
        first, second, third = 0.0, 0.0, 0.0
        denom1 = (knots[i + p] - knots[i]) * (knots[i + p - 1] - knots[i])
        if denom1 != 0:
            first = p * (p - 1) / denom1 * self._basis(i, p - 2, t)
        denom2a = (knots[i + p] - knots[i]) * (knots[i + p] - knots[i + 1])
        denom2b = (knots[i + p + 1] - knots[i + 1]) * (knots[i + p] - knots[i + 1])
        coeff2 = 0.0
        if denom2a != 0:
            coeff2 += 1.0 / denom2a
        if denom2b != 0:
            coeff2 += 1.0 / denom2b
        second = p * (p - 1) * coeff2 * self._basis(i + 1, p - 2, t)
        denom3 = (knots[i + p + 1] - knots[i + 1]) * (knots[i + p + 1] - knots[i + 2])
        if denom3 != 0:
            third = p * (p - 1) / denom3 * self._basis(i + 2, p - 2, t)
        return first - second + third

    def eval(self, t):
        curve = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self._basis(i, self.degree, t)
            curve += N * self.P[i]
        return curve

    def derivative(self, t):
        vec = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self._basis_dt(i, self.degree, t)
            vec += N * self.P[i]
        return vec

    def second_derivative(self, t):
        vec = np.zeros(self.P.shape[1])
        for i in range(len(self.P)):
            N = self._basis_dtt(i, self.degree, t)
            vec += N * self.P[i]
        return vec
