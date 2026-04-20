from src.bspline_curves import *
from src.bspline_optimizers import *
from src.cloud_points import *
from src.visualizers import *
import time
import numpy as np

if __name__ == "__main__":
    
    # problem data
    cloud_points = np.array([[0.0, 0.0],
                         [0.5, 0.5],
                         [1.0, 0.0]])

    initial_guess = [0.0, 0.1, 1.0]
    degree = 2
    knots = np.array([0,0,0,1,1,1])
    Pc = np.array([
        [0.0, 0.0],
        [0.4, 0.0],
        [1.0, 0.0]
    ])

    # curve
    t_vals = np.linspace(0, 1, 100)
    curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #visualize_points(curve)
    #visualize_data_curve_footpoints_controlpoints(cloud_points, curve, np.array([bspline_curve(Pc, t, knots, degree) for t in initial_guess ]), Pc)
    
    # footpoints
    T = all_tk(cloud_points, Pc, knots, degree, initial_guess)

    footpoints = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #visualize_data_curve_footpoints_controlpoints(cloud_points, curve, footpoints, Pc)
    
    optimized_control_points, nb_iter, avg_error = gradient_descent(Pc, T,knots, degree, cloud_points, max_iter=500)
    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    plt.legend("Courbe optimisée")
    #visualize_points(optimized_curve)
    T2 = all_tk(cloud_points, optimized_control_points, knots, degree, T)
    final_footpoints = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    visualize_data_curve_footpoints_controlpoints(cloud_points, optimized_curve, final_footpoints, optimized_control_points)
    visualize_error_convergence(nb_iter, avg_error)
