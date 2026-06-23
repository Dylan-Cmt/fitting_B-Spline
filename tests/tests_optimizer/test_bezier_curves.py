import pytest
from hypothesis import given, strategies as st
from src.optimizers.bezier_curves import *

straight_curve =  np.array([[0.0, 0.0],[0.5, 0.0],[1.0, 0.0]])

@given(n=st.integers(0, 100),t=st.floats(0, 1))
def test_bernstein_basis(n, t):
    values = [bernstein_basis(i, n, t) for i in range(n + 1)]
    # partition of unity
    assert sum(values) == pytest.approx(1.0)
    # positivity
    for b in values:
        assert b >= 0

@given(n=st.integers(0, 100),t=st.floats(0, 1))
def test_bernstein_basis_vector(n, t):
    values = bernstein_basis_vector(n,t)
    # partition of unity
    assert sum(values) == pytest.approx(1.0)
    # positivity
    for b in values:
        assert b >= 0

@given(t=st.floats(0, 1))
def test_eval_bezier_curve(t):
    Pc = straight_curve
    P_t = eval_bezier_curve(Pc, t)
    assert P_t[0] == pytest.approx(t)

def test_eval_bezier_curve_endpoints():
    Pc = straight_curve
    assert np.allclose(eval_bezier_curve(Pc, 0), Pc[0])
    assert np.allclose(eval_bezier_curve(Pc, 1), Pc[-1])

@given(t=st.floats(0, 1))
def test_eval_bezier_derivative(t):
    Pc = np.array([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
    # P_t   = P0 + 2(-P0 + P1)t + (P0 -2P1 +P2)t²
    # P_t'  = 2(-P0 + P1) + 2(P0 -2P1 +P2)t
    dt_P = eval_bezier_derivative(Pc, t)
    true_Pt_prime_x = 0.5 + t
    assert dt_P[0] == pytest.approx(true_Pt_prime_x)

@given(t=st.floats(0, 1))
def test_eval_bezier_second_derivative(t):
    Pc = np.array([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
    # P_t   = P0 + 2(-P0 + P1)t + (P0 -2P1 +P2)t²
    # P_t'  = 2(-P0 + P1) + 2(P0 -2P1 +P2)t
    # P_t'' = 2(P0 -2P1 +P2)
    dtt_P = eval_bezier_second_derivative(Pc, t)
    assert dtt_P[0] == pytest.approx(1)

def test_degree_elevate():
    # null degree returns the same curve
    Pc = np.random.uniform(0, 1, size=(1, 2))
    elevated_Pc = degree_elevate(Pc)
    assert np.allclose(elevated_Pc[-1], Pc[0])
    assert elevated_Pc.shape[0] == Pc.shape[0] + 1

    # degree elevation of a linear curve returns a quadratic curve
    Pc = np.array([[0.0, 0.0], [1.0, 1.0]])
    elevated_Pc = degree_elevate(Pc)
    assert elevated_Pc.shape[0] == 3
    assert np.allclose(elevated_Pc[0], Pc[0])
    assert np.allclose(elevated_Pc[1], (Pc[0] + Pc[1]) / 2)
    assert np.allclose(elevated_Pc[2], Pc[-1])

def test_degree_elevate_multiple():
    # compare all elements to Pc (null degree) and see if its all the same
    Pc = np.random.uniform(0, 1, size=(1, 2))
    for k in range(5):
        elevated_Pc = degree_elevate_multiple(Pc, k)
        for i in range(k):
            assert np.allclose(elevated_Pc[i], Pc[0])
        assert elevated_Pc.shape[0] == Pc.shape[0] + k

    # degree elevation of a linear curve returns a quadratic curve
    Pc = np.array([[0.0, 0.0], [1.0, 0.0]])
    elevated_Pc = degree_elevate_multiple(Pc, 1)
    assert elevated_Pc.shape[0] == 3
    assert np.allclose(elevated_Pc[0], Pc[0])
    assert np.allclose(elevated_Pc[1], (Pc[0] + Pc[1]) / 2)
    assert np.allclose(elevated_Pc[2], Pc[-1])
    elevated_2_Pc = degree_elevate_multiple(Pc, 2)
    assert elevated_2_Pc.shape[0] == 4
    assert np.allclose(elevated_2_Pc[0], Pc[0])
    assert np.allclose(elevated_2_Pc[1], (Pc[0] * 2/3 + Pc[1]) * 1/3)
    assert np.allclose(elevated_2_Pc[2], (Pc[0] * 1/3 + Pc[1]) * 2/3)
    assert np.allclose(elevated_2_Pc[0], Pc[0])

@given(t=st.floats(0, 1))
def test_unit_tangent_norm(t):
    # unit property
    Pc = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    unit_T = unit_tangent(Pc, t)
    norm1 = np.linalg.norm(unit_T)
    assert norm1 == pytest.approx(1.0, abs=1e-7)

    # random curve
    # P_t'  = 2(-P0 + P1) + 2(P0 -2P1 +P2)t = (1, 2-4t)
    dt_P_true = np.array([[1.0, 2 - 4*t]])
    denom2 = np.sqrt(1 + (2 - 4*t)**2)
    u_T_true = dt_P_true / denom2
    assert np.allclose(u_T_true, unit_T)
    

@given(t=st.floats(0, 1))
def test_unit_normal_norm(t):
    # unit property
    Pc = straight_curve
    unit_N = unit_normal(Pc, t)
    norm = np.linalg.norm(unit_N)
    assert norm == pytest.approx(1.0, abs=1e-7)

    # orthogonality to T
    unit_T = unit_tangent(Pc, t)
    assert np.dot(unit_N, unit_T) == pytest.approx(0.0, abs=1e-7)

@given(t=st.floats(0, 1))
def test_curvature2D(t):
    # straight curve
    Pc1 = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    k1 = curvature2D(Pc1, t)
    assert k1 == pytest.approx(0.0, abs=1e-7)

    # random curve
    Pc2 = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    k2 = curvature2D(Pc2, t)
    # P_t'  = 2(-P0 + P1) + 2(P0 -2P1 +P2)t = (1, 2-4t)
    # P_t'' = 2(P0 -2P1 +P2) = (0, -4)
    denom = np.sqrt(1 + (2 - 4*t)**2)
    K_true = 4 / denom**3
    assert k2 == pytest.approx(K_true, abs=1e-7)

@given(t=st.floats(0,1))
def test_curvature_radius(t):
    # straight curve
    Pc = straight_curve
    rho = curvature_radius(Pc, t)
    assert rho == pytest.approx(1e12, abs=1e-7)
