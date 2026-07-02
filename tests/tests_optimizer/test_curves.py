import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.optimizers.curves import Curve, BezierCurve, BSplineCurve


# -----------------------------
# Global / Curve (abstract) tests
# -----------------------------

def test_curve_abstract_methods_raise():
    c = Curve()
    with pytest.raises(NotImplementedError):
        c.basis(0, 0, 0.5)
    with pytest.raises(NotImplementedError):
        c.eval(0.5)
    with pytest.raises(NotImplementedError):
        c.derivative(0.5)
    with pytest.raises(NotImplementedError):
        c.second_derivative(0.5)
    with pytest.raises(NotImplementedError):
        c.degree_elevate()
    with pytest.raises(NotImplementedError):
        c.degree_elevate_multiple(1)


# -----------------------------
# BezierCurve tests (ordered)
# -----------------------------

# simple straight curve used across tests
STRAIGHT = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])


@given(n=st.integers(0, 20), t=st.floats(0.0, 1.0))
def test_bernstein_basis_partition_and_positivity(n, t):
    # use the instance bernstein via temporary BezierCurve
    # if n == 0 we still want a valid curve, use single control point
    ctrl = np.zeros((max(1, n + 1), 2))
    B = BezierCurve(ctrl)
    values = [B.basis(i, n, t) for i in range(n + 1)]
    assert sum(values) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in values)


@given(n=st.integers(0, 20), t=st.floats(0.0, 1.0))
def test__bernstein_basis_vector_partition(n, t):
    ctrl = np.zeros((max(1, n + 1), 2))
    B = BezierCurve(ctrl)
    vec = B._bernstein_basis_vector(n, t)
    assert vec.shape[0] == n + 1
    assert sum(vec) == pytest.approx(1.0)


@given(t=st.floats(0.0, 1.0))
def test_eval_bezier_curve_on_straight_line(t):
    bc = BezierCurve(STRAIGHT)
    pt = bc.eval(t)
    # x coordinate should be equal to t on this parametrization
    assert pt[0] == pytest.approx(t)


def test_eval_bezier_curve_endpoints():
    bc = BezierCurve(STRAIGHT)
    assert np.allclose(bc.eval(0.0), STRAIGHT[0])
    assert np.allclose(bc.eval(1.0), STRAIGHT[-1])


@given(t=st.floats(0.0, 1.0))
def test_bezier_first_derivative_matches_polynomial(t):
    Pc = np.array([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
    bc = BezierCurve(Pc)
    dt = bc.derivative(t)
    # Expected derivative x component: 0.5 + t
    assert dt[0] == pytest.approx(0.5 + t)


@given(t=st.floats(0.0, 1.0))
def test_bezier_second_derivative_matches_polynomial(t):
    Pc = np.array([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
    bc = BezierCurve(Pc)
    dtt = bc.second_derivative(t)
    assert dtt[0] == pytest.approx(1.0)


def test_bezier_degree_elevate_constant_curve():
    Pc = np.random.RandomState(0).uniform(0, 1, size=(1, 2))
    bc = BezierCurve(Pc)
    elevated = bc.degree_elevate()
    # shape increases by exactly 1
    assert elevated.shape[0] == Pc.shape[0] + 1
    # last control point equals original (constant curve)
    assert np.allclose(elevated[-1], Pc[0])


def test_bezier_degree_elevate_linear_to_quadratic():
    Pc = np.array([[0.0, 0.0], [1.0, 1.0]])
    bc = BezierCurve(Pc)
    elevated = bc.degree_elevate()
    assert elevated.shape[0] == 3
    assert np.allclose(elevated[0], Pc[0])
    assert np.allclose(elevated[1], (Pc[0] + Pc[1]) / 2)
    assert np.allclose(elevated[2], Pc[-1])


def test_bezier_degree_elevate_multiple():
    Pc = np.array([[0.0, 0.0], [1.0, 0.0]])
    bc = BezierCurve(Pc)
    elevated1 = bc.degree_elevate_multiple(1)
    assert elevated1.shape[0] == 3
    elevated2 = bc.degree_elevate_multiple(2)
    assert elevated2.shape[0] == 4


@given(t=st.floats(0.0, 1.0))
def test_bezier_unit_tangent_and_normal(t):
    Pc = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    bc = BezierCurve(Pc)
    uT = bc.unit_tangent(t)
    assert np.linalg.norm(uT) == pytest.approx(1.0, abs=1e-7)
    uN = bc.unit_normal(t)
    assert np.linalg.norm(uN) == pytest.approx(1.0, abs=1e-7)
    assert abs(np.dot(uN, uT)) == pytest.approx(0.0, abs=1e-7)


@given(t=st.floats(0.0, 1.0))
def test_bezier_curvature_and_radius(t):
    # straight curve -> curvature 0 and radius large
    bc = BezierCurve(STRAIGHT)
    k = bc.curvature2D(t)
    assert k == pytest.approx(0.0, abs=1e-7)
    rho = bc.curvature_radius(t)
    assert rho == pytest.approx(1e12, abs=1e-7)


# -----------------------------
# BSplineCurve tests (ordered)
# -----------------------------

STRAIGHT_B = np.linspace(0, 1, 10)[:, np.newaxis]
STRAIGHT_B = np.hstack((STRAIGHT_B, np.zeros_like(STRAIGHT_B)))
CLOSED = np.array([[0.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]])


def _knots_for(n_ctrl, degree):
    # uniform clamped knots in [0,1]
    internal = np.linspace(0.0, 1.0, n_ctrl - degree + 1)
    knots = np.concatenate(([internal[0]] * degree, internal, [internal[-1]] * degree))
    return knots


@given(degree=st.integers(0, 5), t=st.floats(0.0, 1.0))
def test_bspline_basis_partition_of_unity_and_positivity(degree, t):
    n = len(STRAIGHT_B)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(STRAIGHT_B, knots, degree)
    values = [bsp.basis(i, degree, t) for i in range(n)]
    assert sum(values) == pytest.approx(1.0)
    assert all(v >= -1e-12 for v in values)


@given(t=st.floats(0.0, 1.0, exclude_max=True))
def test_bspline_basis_exact_values_nurbs_book(t):
    # Example 2.1 p.52 from The NURBS Book
    knots = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    bsp = BSplineCurve(np.zeros((3, 2)), knots, 2)
    # N_{0,0}=0, N_{1,0}=0, N_{2,0}=1, N_{3,0}=0, N_{4,0}=0
    assert bsp.basis(0, 0, t) == pytest.approx(0.0)
    assert bsp.basis(1, 0, t) == pytest.approx(0.0)
    assert bsp.basis(2, 0, t) == pytest.approx(1.0)
    assert bsp.basis(3, 0, t) == pytest.approx(0.0)
    assert bsp.basis(4, 0, t) == pytest.approx(0.0)
    # degree 1
    assert bsp.basis(0, 1, t) == pytest.approx(0.0)
    assert bsp.basis(1, 1, t) == pytest.approx(1 - t)
    assert bsp.basis(2, 1, t) == pytest.approx(t)
    assert bsp.basis(3, 1, t) == pytest.approx(0.0)
    # degree 2
    assert bsp.basis(0, 2, t) == pytest.approx((1 - t) ** 2)
    assert bsp.basis(1, 2, t) == pytest.approx(2 * t * (1 - t))
    assert bsp.basis(2, 2, t) == pytest.approx(t ** 2)


@given(degree=st.integers(0, 5), t=st.floats(0.0, 1.0))
def test_bspline_basis_derivative_partition_of_unity(degree, t):
    n = len(STRAIGHT_B)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(STRAIGHT_B, knots, degree)
    # first derivative sum must be zero
    values_dt = [bsp._basis_dt(i, degree, t) for i in range(n)]
    assert sum(values_dt) == pytest.approx(0.0)
    # second derivative sum must be zero
    values_dtt = [bsp._basis_dtt(i, degree, t) for i in range(n)]
    assert sum(values_dtt) == pytest.approx(0.0)


def _bspline_hodograph(ctrlpts, knots, degree):
    Q = []
    for i in range(len(ctrlpts) - 1):
        denom = knots[i + degree + 1] - knots[i + 1]
        if np.isclose(denom, 0.0):
            Q.append(np.zeros_like(ctrlpts[i]))
        else:
            Q.append(degree * (ctrlpts[i + 1] - ctrlpts[i]) / denom)
    return np.array(Q)


@given(degree=st.integers(0, 4), t=st.floats(0.0, 1.0))
def test_bspline_curve_eval_and_monotony(degree, t):
    n = len(STRAIGHT_B)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(STRAIGHT_B, knots, degree)
    P_t = bsp.eval(t)
    # x coordinate equals convex combination of control x's
    xs = [bsp.basis(i, degree, t) * (i / (n - 1)) for i in range(n)]
    assert P_t[0] == pytest.approx(sum(xs))
    assert P_t[1] == pytest.approx(0.0)
    # monotony on straight line
    h = 1e-6
    if t + h <= 1.0:
        P_th = bsp.eval(t + h)
        assert P_th[0] >= P_t[0]


@given(degree=st.integers(0, 4))
def test_bspline_eval_endpoints(degree):
    n = len(STRAIGHT_B)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(STRAIGHT_B, knots, degree)
    assert np.allclose(bsp.eval(0.0), STRAIGHT_B[0])
    assert np.allclose(bsp.eval(1.0), STRAIGHT_B[-1])


@given(degree=st.integers(0, 4), t=st.floats(0.0, 1.0))
def test_bspline_first_and_second_derivative_against_hodograph(degree, t):
    n = len(CLOSED)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(CLOSED, knots, degree)
    dt_P = bsp.derivative(t)
    if degree < 1:
        dt_ref = np.zeros_like(CLOSED[0])
    else:
        deriv_ctrlpts = _bspline_hodograph(CLOSED, knots, degree)
        inner_knots = knots[1:-1]
        dt_ref = BSplineCurve(deriv_ctrlpts, inner_knots, degree - 1).eval(t)
    assert np.allclose(dt_P, dt_ref)

    dtt_P = bsp.second_derivative(t)
    if degree < 2:
        dtt_ref = np.zeros_like(CLOSED[0])
    else:
        first_deriv_ctrlpts = _bspline_hodograph(CLOSED, knots, degree)
        second_deriv_ctrlpts = _bspline_hodograph(first_deriv_ctrlpts, knots[1:-1], degree - 1)
        inner_knots_2 = knots[2:-2]
        dtt_ref = BSplineCurve(second_deriv_ctrlpts, inner_knots_2, degree - 2).eval(t)
    assert np.allclose(dtt_P, dtt_ref)


@given(degree=st.integers(0, 4), t=st.floats(0.0, 1.0))
def test_bspline_unit_tangent_normal_and_curvature(degree, t):
    n = len(STRAIGHT_B)
    knots = _knots_for(n, degree)
    bsp = BSplineCurve(STRAIGHT_B, knots, degree)
    if degree < 1:
        with pytest.raises(ValueError):
            bsp.unit_tangent(t)
        return

    uT = bsp.unit_tangent(t)
    assert np.isclose(np.linalg.norm(uT), 1.0)
    uN = bsp.unit_normal(t)
    assert np.isclose(np.linalg.norm(uN), 1.0)
    assert np.isclose(np.dot(uN, uT), 0.0)
    # straight line special case curvature ~ 0
    K = bsp.curvature2D(t)
    assert K == pytest.approx(0.0)
    rho = bsp.curvature_radius(t)
    assert rho == pytest.approx(1e12)
