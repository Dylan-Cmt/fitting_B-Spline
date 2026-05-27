import pytest
from hypothesis import given, strategies as st
from src.bspline_curves import *

straight_curve = np.linspace(0,1,10)[:, np.newaxis]
straight_curve = np.hstack((straight_curve, np.zeros_like(straight_curve)))
closed_curve = np.array([[0.0, -1.0],[-1.0, -1.0],[-1.0,  1.0],[1.0,  1.0],[1.0, -1.0],[0.0, -1.0]])
# knot size = nb_ctrlpts + degree + 1
n = len(straight_curve) # len(closed_curve)
n_closed = len(closed_curve)


# Do not try high degree, test could take exp time or even fail
@given(degree=st.integers(0, n-1),t=st.floats(0, 1))
def test_bspline_basis_properties(degree,t):
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    values = [bspline_basis(i, degree, t, knots) for i in range(n)]
    # Partition of unity
    assert sum(values) == pytest.approx(1.0)
    # Positivity
    for b in values:
        assert b >= 0

# Values are from Example2.1 p52 of the NURBS book
@given(t=st.floats(0, 1,exclude_max=True))
def test_bspline_basis(t):
    knots = np.array([0,0,0,1,1,1])

    # N_{0,0}(t) = 0
    assert bspline_basis(0, 0, t, knots) == pytest.approx(0.0)
    # N_{1,0}(t) = 0
    assert bspline_basis(1, 0, t, knots) == pytest.approx(0.0)
    # N_{2,0}(t) = 1
    assert bspline_basis(2, 0, t, knots) == pytest.approx(1.0)
    # N_{3,0}(t) = 0
    assert bspline_basis(3, 0, t, knots) == pytest.approx(0.0)
    # N_{4,0}(t) = 0
    assert bspline_basis(4, 0, t, knots) == pytest.approx(0.0)

    # N_{0,1}(t) = 0
    assert bspline_basis(0, 1, t, knots) == pytest.approx(0.0)
    # N_{1,1}(t) = 1 - t
    assert bspline_basis(1, 1, t, knots) == pytest.approx(1 - t)
    # N_{2,1}(t) = t
    assert bspline_basis(2, 1, t, knots) == pytest.approx(t)
    # N_{3,1}(t) = 0
    assert bspline_basis(3, 1, t, knots) == pytest.approx(0.0)

    # N_{0,2}(t) = (1 - t)^2
    assert bspline_basis(0, 2, t, knots) == pytest.approx((1 - t)**2)
    # N_{1,2}(t) = 2t(1 - t)
    assert bspline_basis(1, 2, t, knots) == pytest.approx(2 * t * (1 - t))
    # N_{2,2}(t) = t^2
    assert bspline_basis(2, 2, t, knots) == pytest.approx(t**2)
    
@given(degree=st.integers(0, n-1),t=st.floats(0, 1))
def test_dt_bspline_basis(degree,t):
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    values = [dt_bspline_basis(i, degree, t, knots) for i in range(n)]
    # Derivative of the partition of unity
    assert sum(values) == pytest.approx(0.0)

@given(degree=st.integers(0, n-1),t=st.floats(0, 1))
def test_dtt_bspline_basis(degree,t):
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    values = [dtt_bspline_basis(i, degree, t, knots) for i in range(n)]
    # Second derivative of the partition of unity
    assert sum(values) == pytest.approx(0.0)


@given(degree=st.integers(0, n-1),t=st.floats(0, 1))
def test_bspline_curve(degree,t):
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    P_t = bspline_curve(straight_curve, t, knots, degree)
    assert P_t[0] == pytest.approx(sum([bspline_basis(i, degree, t, knots)*i/(n-1) for i in range(n)]))
    assert P_t[1] == pytest.approx(0.0)

    # Monotony
    h = 1e-6
    if t + h <= 1 :
        P_t_plus_h = bspline_curve(straight_curve, t + h, knots, degree)
        assert P_t_plus_h[0] >= P_t[0]

@given(degree=st.integers(0, n-1))
def test_eval_bezier_curve_endpoints(degree):
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    assert np.allclose(bspline_curve(straight_curve, 0, knots, degree), straight_curve[0])
    assert np.allclose(bspline_curve(straight_curve, 1, knots, degree), straight_curve[-1])

def hodograph(ctrlpts, knots, degree):
    Q = []
    for i in range(len(ctrlpts) - 1):
        denom = knots[i + degree + 1] - knots[i + 1]
        if np.isclose(denom, 0.0):
            Q_i = np.zeros_like(ctrlpts[i])
        else:
            Q_i = degree * (ctrlpts[i + 1] - ctrlpts[i]) / denom
        Q.append(Q_i)
    return np.array(Q)

@given(degree=st.integers(0, n_closed-1), t=st.floats(0, 1))
def test_dt_bspline_curve(degree, t):
    knots = np.arange(n_closed - degree + 1) / (n_closed -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # Derivative to test
    dt_P = dt_bspline_curve(closed_curve, t, knots, degree)

    # Derivative obtained with the hodograph
    if degree < 1:
        dt_P_ref = np.zeros_like(closed_curve[0])
    else:
        deriv_ctrlpts = hodograph(closed_curve,knots,degree)
        dt_P_ref = bspline_curve(deriv_ctrlpts,t,knots[1:-1], degree - 1)
    assert np.allclose(dt_P, dt_P_ref)

@given(degree=st.integers(0, n_closed-1), t=st.floats(0, 1))
def test_dtt_bspline_curve(degree, t):
    knots = np.arange(n_closed - degree + 1) / (n_closed - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # Second derivative to test
    dtt_P = dtt_bspline_curve(closed_curve, t, knots, degree)

    # Second derivative obtained with the hodograph
    if degree < 2:
        dtt_P_ref = np.zeros_like(closed_curve[0])
    else:
        first_deriv_ctrlpts = hodograph(closed_curve, knots, degree)
        second_deriv_ctrlpts = hodograph(first_deriv_ctrlpts, knots[1:-1], degree - 1)

        dtt_P_ref = bspline_curve(second_deriv_ctrlpts, t, knots[2:-2], degree - 2)

    assert np.allclose(dtt_P, dtt_P_ref)

@given(degree=st.integers(0, n-1), t=st.floats(0, 1))
def test_unit_tangent(degree, t):
    knots = np.arange(n - degree + 1) / (n - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    if degree < 1:
        with pytest.raises(ValueError):
            unit_tangent(straight_curve, t, knots, degree)
        return

    unit_T = unit_tangent(straight_curve, t, knots, degree)

    assert np.dot(unit_T, (0,1)) == pytest.approx(0.0)

@given(degree=st.integers(0, n_closed-1), t=st.floats(0, 1))
def test_unit_tangent_norm(degree, t):
    knots = np.arange(n_closed - degree + 1) / (n_closed - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    if degree < 1:
        with pytest.raises(ValueError):
            unit_tangent(closed_curve, t, knots, degree)
        return
        
    unit_T = unit_tangent(closed_curve, t, knots, degree)
    norm1 = np.linalg.norm(unit_T)
    assert norm1 == pytest.approx(1.0, abs=1e-7)

@given(degree=st.integers(0, n_closed-1), t=st.floats(0, 1))
def test_unit_normal(degree, t):
    knots = np.arange(n_closed - degree + 1) / (n_closed - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    if degree < 1:
        with pytest.raises(ValueError):
            unit_normal(closed_curve, t, knots, degree)
        return

    unit_T = unit_tangent(closed_curve, t, knots, degree)
    unit_N = unit_normal(closed_curve, t, knots, degree)

    assert np.linalg.norm(unit_N) == pytest.approx(1.0, abs=1e-7)
    assert np.dot(unit_N, unit_T) == pytest.approx(0.0, abs=1e-7)

@given(degree=st.integers(0, n-1), t=st.floats(0, 1))
def test_curvature2D(degree, t):
    knots = np.arange(n - degree + 1) / (n - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    if degree < 1:
        with pytest.raises(ValueError):
            curvature2D(straight_curve , t, knots, degree)
        return

    K = curvature2D(straight_curve , t, knots, degree)
    assert K == pytest.approx(0.0)

@given(degree=st.integers(0, n-1), t=st.floats(0, 1))
def test_curvature_radius(degree, t):
    knots = np.arange(n - degree + 1) / (n - degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    if degree < 1:
        with pytest.raises(ValueError):
            curvature_radius(straight_curve , t, knots, degree)
        return

    rho = curvature_radius(straight_curve , t, knots, degree)
    assert rho == pytest.approx(1e12)