import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.optimizers.curves import BezierCurve, BSplineCurve
from src.optimizers.optimizers import PDM, TDM, SDM


STRAIGHT_BEZIER_POINTS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
STRAIGHT_BSPLINE_POINTS = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
BSPLINE_KNOTS = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=float)
BSPLINE_DEGREE = 1

ON_CURVE_POINTS = np.array([[0.0, 0.0], [0.33, 0.0], [0.66, 0.0], [1.0, 0.0]], dtype=float)
ON_CURVE_T = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)

X1 = np.array([0.1, 0.1], dtype=float)
X2 = np.array([0.1, 0.5], dtype=float)
X3 = np.array([0.1, 1.0], dtype=float)
X4 = np.array([0.5, 0.1], dtype=float)
X5 = np.array([1.0, 0.0], dtype=float)
X6 = np.array([2.5, 0.0], dtype=float)

ALL_DATA_BEZIER = np.array([X1, X2, X3, X4, X5, X6], dtype=float)
T_TRUE_BEZIER = np.array([0.1, 0.1, 0.1, 0.5, 1.0, 1.0], dtype=float)

ALL_DATA_BSPLINE = np.array([X1, X2, X3], dtype=float)
T_TRUE_BSPLINE = np.array([0.1, 0.1, 0.1], dtype=float)

class TestBaseOptimizer:

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_newton_tk_projects_data_points_to_true_parameters(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            all_data = ALL_DATA_BEZIER
            t_true = T_TRUE_BEZIER
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            all_data = ALL_DATA_BSPLINE
            t_true = T_TRUE_BSPLINE
        
        pdm = PDM(curve)
        for Xk, t_ref in zip(all_data, t_true):
            t_opt = pdm.newton_tk(pdm.err_PD_prim, pdm.err_PD_primprim, Xk, initial_guess=0.9)
            assert t_opt == pytest.approx(t_ref, abs=1e-6)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_all_tk_with_and_without_initial_guesses(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            all_data = ALL_DATA_BEZIER
            t_true = T_TRUE_BEZIER
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            all_data = ALL_DATA_BSPLINE
            t_true = T_TRUE_BSPLINE
        
        pdm = PDM(curve)
        computed = pdm.all_tk(all_data)
        assert np.allclose(computed, t_true, atol=1e-6)

        bad_guess = [0.9] * len(all_data)
        computed2 = pdm.all_tk(all_data, initial_guesses=bad_guess)
        assert np.allclose(computed2, t_true, atol=1e-6)


class TestPDM:

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_err_PD_exact_values(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        pdm = PDM(curve)
        
        X_on = np.array([0.1, 0.0])
        assert pdm.err_PD(X_on, 0.1) == pytest.approx(0.0, abs=1e-7)
        
        X_off = np.array([0.1, 0.5])
        assert pdm.err_PD(X_off, 0.1) == pytest.approx(0.5**2, abs=1e-7)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_f_PD_is_zero_for_points_on_curve(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        pdm = PDM(curve)
        assert pdm.f_PD(X, T) == pytest.approx(0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_dP_f_PD_is_zero_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        pdm = PDM(curve)
        gradient = pdm.dP_f_PD(X, T)
        assert gradient.shape == control_points.shape
        assert np.allclose(gradient, 0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @given(values=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_dP_f_PD_returns_valid_gradient(self, curve_type, values):
        if curve_type == "bezier":
            P = np.array(values, dtype=float).reshape(4, 2)
            curve = BezierCurve(P)
            X = P + np.array([0.05, -0.07])
            T = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)
        else:
            P = np.array(values[:6], dtype=float).reshape(3, 2)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = P + np.array([0.05, -0.07])
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        pdm = PDM(curve)
        gradient = pdm.dP_f_PD(X, T)
        assert gradient.shape == P.shape
        assert np.isfinite(gradient).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_phi_PD_and_derivatives_consistency(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = np.array([[0.1, 0.1], [0.4, 0.05], [0.8, 0.2], [0.95, 0.12]], dtype=float)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.1, 0.1], [0.4, 0.05], [0.8, 0.2]], dtype=float)
        
        pdm = PDM(curve)
        T = pdm.all_tk(X)
        alpha = 0.03
        eps = 1e-6
        phi = pdm.phi_PD(alpha, X, T)
        dphi = pdm.d_phi_PD(alpha, X, T)
        phi_plus = pdm.phi_PD(alpha + eps, X, T)
        phi_minus = pdm.phi_PD(alpha - eps, X, T)
        finite_diff = (phi_plus - phi_minus) / (2.0 * eps)
        assert np.isfinite(dphi)
        assert dphi == pytest.approx(finite_diff, rel=1e-3, abs=1e-4)

        ddphi = pdm.dd_phi_PD(alpha, X, T)
        finite_diff_second = (phi_plus - 2.0 * phi + phi_minus) / (eps**2)
        assert np.isfinite(ddphi)
        assert ddphi == pytest.approx(finite_diff_second, rel=1e-3, abs=1e-4)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_PD_returns_zero_iterations_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        pdm = PDM(curve)
        P_new, log_iter, log_avg_error, log_max_error = pdm.gradient_descent_PD(X, T, alpha0=0.05, max_iter=10, tol=1e-8, constraint="opened")
        assert np.allclose(P_new, control_points)
        assert 10**log_iter[-1] == 1
        assert 10**log_avg_error[-1] == 0
        assert 10**log_max_error[-1] == 0

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_PD_reduces_avg_and_max_error(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        pdm = PDM(curve)
        T0 = pdm.all_tk(X)
        P_new, log_iter, log_avg_error, log_max_error = pdm.gradient_descent_PD(X, T0, alpha0=0.05, max_iter=10, tol=1e-6, constraint="opened")
        assert P_new.shape == P.shape
        assert len(log_iter) == len(log_avg_error) == len(log_max_error)
        assert log_avg_error[0] > log_avg_error[-1]
        assert log_max_error[0] > log_max_error[-1]
        assert np.isfinite(P_new).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_PD_respects_opened_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        pdm = PDM(curve)
        T0 = pdm.all_tk(X)
        P_new, _, _, _ = pdm.gradient_descent_PD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="opened")
        assert np.allclose(P_new[0], P[0])
        assert np.allclose(P_new[-1], P[-1])

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_PD_respects_closed_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 0.9], [0.75, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.8], [0.5, 1.0], [0.75, 0.8], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.8], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        pdm = PDM(curve)
        T0 = pdm.all_tk(X)
        P_new, _, _, _ = pdm.gradient_descent_PD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="closed")
        assert np.allclose(P_new[-1], P_new[0])

class TestTDM:

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_err_TD_is_zero_for_points_on_curve(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        tdm = TDM(curve)
        assert tdm.err_TD(np.array([0.1, 0.0]), 0.1) == pytest.approx(0.0)
        assert tdm.err_TD(np.array([0.5, 0.0]), 0.5) == pytest.approx(0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_f_TD_is_zero_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        tdm = TDM(curve)
        assert tdm.f_TD(X, T) == pytest.approx(0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_dP_f_TD_is_zero_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        tdm = TDM(curve)
        gradient = tdm.dP_f_TD(X, T)
        assert gradient.shape == control_points.shape
        assert np.allclose(gradient, 0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @given(values=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_dP_f_TD_returns_valid_gradient(self, curve_type, values):
        if curve_type == "bezier":
            P = np.array(values, dtype=float).reshape(4, 2)
            curve = BezierCurve(P)
            X = P + np.array([0.0, 0.1])
            T = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)
        else:
            P = np.array(values[:6], dtype=float).reshape(3, 2)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = P + np.array([0.0, 0.1])
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        tdm = TDM(curve)
        gradient = tdm.dP_f_TD(X, T)
        assert gradient.shape == P.shape
        assert np.isfinite(gradient).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_phi_TD_and_derivatives_consistency(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = np.array([[0.1, 0.2], [0.4, 0.05], [0.75, 0.2], [0.95, 0.15]], dtype=float)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.1, 0.2], [0.4, 0.05], [0.75, 0.2]], dtype=float)
        
        tdm = TDM(curve)
        T = tdm.all_tk(X)
        alpha = 0.02
        eps = 1e-6
        phi = tdm.phi_TD(alpha, X, T)
        dphi = tdm.d_phi_TD(alpha, X, T)
        phi_plus = tdm.phi_TD(alpha + eps, X, T)
        phi_minus = tdm.phi_TD(alpha - eps, X, T)
        finite_diff = (phi_plus - phi_minus) / (2.0 * eps)
        assert np.isfinite(dphi)
        assert dphi == pytest.approx(finite_diff, rel=1e-3, abs=1e-5)

        ddphi = tdm.dd_phi_TD(alpha, X, T)
        finite_diff_second = (phi_plus - 2.0 * phi + phi_minus) / (eps**2)
        assert np.isfinite(ddphi)
        assert ddphi == pytest.approx(finite_diff_second, rel=1e-3, abs=1e-3)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_TD_returns_zero_iterations_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        tdm = TDM(curve)
        P_new, log_iter, log_avg_error, log_max_error = tdm.gradient_descent_TD(X, T, alpha0=0.05, max_iter=10, tol=1e-8, constraint="opened")
        assert np.allclose(P_new, control_points)
        assert 10**log_iter[-1] == 1
        assert 10**log_avg_error[-1] == 0
        assert 10**log_max_error[-1] == 0

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_TD_reduces_avg_and_max_error(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        tdm = TDM(curve)
        T0 = tdm.all_tk(X)
        P_new, log_iter, log_avg_error, log_max_error = tdm.gradient_descent_TD(X, T0, alpha0=0.05, max_iter=10, tol=1e-6, constraint="opened")
        assert P_new.shape == P.shape
        assert len(log_iter) == len(log_avg_error) == len(log_max_error)
        assert log_avg_error[0] > log_avg_error[-1]
        assert log_max_error[0] > log_max_error[-1]
        assert np.isfinite(P_new).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_TD_respects_opened_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        tdm = TDM(curve)
        T0 = tdm.all_tk(X)
        P_new, _, _, _ = tdm.gradient_descent_TD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="opened")
        assert np.allclose(P_new[0], P[0])
        assert np.allclose(P_new[-1], P[-1])

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_TD_respects_closed_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 0.9], [0.75, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.8], [0.5, 1.0], [0.75, 0.8], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.8], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        tdm = TDM(curve)
        T0 = tdm.all_tk(X)
        P_new, _, _, _ = tdm.gradient_descent_TD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="closed")
        assert np.allclose(P_new[-1], P_new[0])


class TestSDM:

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_signed_distance_is_zero_for_on_curve_points(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            points = ON_CURVE_POINTS
            ts = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            ts = np.array([0.0, 0.5, 1.0], dtype=float)
        
        sdm = SDM(curve)
        for point, t in zip(points, ts):
            assert sdm.signed_distance(point, t) == pytest.approx(0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_err_SD_matches_err_TD_for_positive_signed_distance(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = np.array([0.5, 1.0], dtype=float)
            t = 0.5
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([0.5, 1.0], dtype=float)
            t = 0.5
        
        sdm = SDM(curve)
        tdm = TDM(curve)
        error_tdm = tdm.err_TD(X, t)
        assert sdm.err_SD(X, t) == pytest.approx(error_tdm)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_err_SD_is_nonnegative_for_off_curve_points(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        sdm = SDM(curve)
        X = np.array([0.5, -1.0], dtype=float)
        assert sdm.err_SD(X, 0.5) >= 0.0

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_f_SD_is_zero_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        sdm = SDM(curve)
        assert sdm.f_SD(X, T) == pytest.approx(0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_dP_f_SD_is_zero_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        sdm = SDM(curve)
        gradient = sdm.dP_f_SD(X, T)
        assert gradient.shape == control_points.shape
        assert np.allclose(gradient, 0.0)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @given(values=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_dP_f_SD_returns_valid_gradient(self, curve_type, values):
        if curve_type == "bezier":
            P = np.array(values, dtype=float).reshape(4, 2)
            curve = BezierCurve(P)
            X = P + np.array([0.0, 0.1])
            T = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)
        else:
            P = np.array(values[:6], dtype=float).reshape(3, 2)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = P + np.array([0.0, 0.1])
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        sdm = SDM(curve)
        gradient = sdm.dP_f_SD(X, T)
        assert gradient.shape == P.shape
        assert np.isfinite(gradient).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    def test_phi_SD_and_derivatives_consistency(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            X = np.array([[0.1, 0.2], [0.35, 0.1], [0.7, 0.2], [0.95, 0.05]], dtype=float)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            X = np.array([[0.1, 0.2], [0.35, 0.1], [0.7, 0.2]], dtype=float)

        sdm = SDM(curve)
        T = sdm.all_tk(X)
        alpha = 0.02
        eps = 1e-6
        phi = sdm.phi_SD(alpha, X, T)
        dphi = sdm.d_phi_SD(alpha, X, T)
        phi_plus = sdm.phi_SD(alpha + eps, X, T)
        phi_minus = sdm.phi_SD(alpha - eps, X, T)
        finite_diff = (phi_plus - phi_minus) / (2.0 * eps)
        assert np.isfinite(dphi)
        assert dphi == pytest.approx(finite_diff, rel=1e-3, abs=1e-5)

        ddphi = sdm.dd_phi_SD(alpha, X, T)
        finite_diff_second = (phi_plus - 2.0 * phi + phi_minus) / (eps**2)
        assert np.isfinite(ddphi)
        assert ddphi == pytest.approx(finite_diff_second, rel=1e-3, abs=1e-2)

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_SD_returns_zero_iterations_for_on_curve_data(self, curve_type):
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
            control_points = STRAIGHT_BEZIER_POINTS
            X = ON_CURVE_POINTS
            T = ON_CURVE_T
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            control_points = STRAIGHT_BSPLINE_POINTS
            X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
            T = np.array([0.0, 0.5, 1.0], dtype=float)
        
        sdm = SDM(curve)
        P_new, log_iter, log_avg_error, log_max_error = sdm.gradient_descent_SD(X, T, alpha0=0.05, max_iter=10, tol=1e-8, constraint="opened")
        assert np.allclose(P_new, control_points)
        assert 10**log_iter[-1] == 1
        assert 10**log_avg_error[-1] == 0
        assert 10**log_max_error[-1] == 0

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_SD_reduces_avg_and_max_error(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        sdm = SDM(curve)
        T0 = sdm.all_tk(X)
        P_new, log_iter, log_avg_error, log_max_error = sdm.gradient_descent_SD(X, T0, alpha0=0.05, max_iter=10, tol=1e-6, constraint="opened")
        assert P_new.shape == P.shape
        assert len(log_iter) == len(log_avg_error) == len(log_max_error)
        assert log_avg_error[0] > log_avg_error[-1]
        assert log_max_error[0] > log_max_error[-1]
        assert np.isfinite(P_new).all()

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_SD_respects_opened_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.75, 0.75], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        sdm = SDM(curve)
        T0 = sdm.all_tk(X)
        P_new, _, _, _ = sdm.gradient_descent_SD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="opened")
        assert np.allclose(P_new[0], P[0])
        assert np.allclose(P_new[-1], P[-1])

    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_SD_respects_closed_constraint(self, curve_type):
        if curve_type == "bezier":
            P = np.array([[0.0, 0.0], [0.25, 0.9], [0.75, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.25, 0.8], [0.5, 1.0], [0.75, 0.8], [1.0, 0.0]], dtype=float)
            curve = BezierCurve(P)
        else:
            P = np.array([[0.0, 0.0], [0.5, 0.9], [1.0, 0.0]], dtype=float)
            X = np.array([[0.0, 0.0], [0.5, 0.8], [1.0, 0.0]], dtype=float)
            curve = BSplineCurve(P, BSPLINE_KNOTS, BSPLINE_DEGREE)
        
        sdm = SDM(curve)
        T0 = sdm.all_tk(X)
        P_new, _, _, _ = sdm.gradient_descent_SD(X, T0, alpha0=0.05, max_iter=5, tol=1e-8, constraint="closed")
        assert np.allclose(P_new[-1], P_new[0])
    
    @pytest.mark.parametrize("curve_type", ["bezier", "bspline"])
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_gradient_descent_SD_data_already_on_curve(self, curve_type):
        """Vérifie que l'optimiseur s'arrête immédiatement si les données coïncident déjà (Zéro itération utile)."""
        if curve_type == "bezier":
            curve = BezierCurve(STRAIGHT_BEZIER_POINTS)
        else:
            curve = BSplineCurve(STRAIGHT_BSPLINE_POINTS, BSPLINE_KNOTS, BSPLINE_DEGREE)
            
        sdm = SDM(curve)
        
        P_new, log_iter, log_avg_error, log_max_error = sdm.gradient_descent_SD(
            ON_CURVE_POINTS, ON_CURVE_T, alpha0=0.1, max_iter=100, tol=1e-6, constraint="opened"
        )
        
        assert np.allclose(P_new, curve.P)
        assert 10**log_iter[-1] == pytest.approx(1.0, abs=1e-5)
        assert 10**log_avg_error[-1] == pytest.approx(0.0, abs=1e-5)
        assert 10**log_max_error[-1] == pytest.approx(0.0, abs=1e-5)