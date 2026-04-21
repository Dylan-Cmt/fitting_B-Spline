from src.bezier_curves import *
from src.bezier_optimizer import *
from src.cloud_points import *
from src.visualizers import *
import time

if __name__ == "__main__":
    
    #################################################
    #                                               #
    #               simple example                  #
    #                                               #
    #################################################

    """
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    control_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    tk_initial_guess = [0.0, 0.1, 1.0]
    """
    """
    #X = np.array([[0.0, 0.0], [0.4, 0.6], [0.6, 0.4], [1.0, 0.0]])
    X = generate_sinus_cloud_points(50, 0)
    #X = generate_square_cloud_points(100, 0.04)
    #control_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    #control_points = np.array([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.0, 0.0]])
    control_points = np.linspace(0,1,3)[:, np.newaxis]
    control_points = np.hstack((control_points, np.zeros_like(control_points)))
    #tk_initial_guess = [0.0, 0.1, 0.3, 1.0]
    tk_initial_guess = np.linspace(0.0, 1.0, len(X))
    
    
    unoptimized_curve = eval_bezier_curve(control_points, np.linspace(0.0, 1.0, 50))

    plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(control_points)-1}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(control_points, t) for t in tk_initial_guess], control_points)

    T = all_tk(X, control_points, initial_guesses=tk_initial_guess)

    plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(control_points)-1}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(control_points, t) for t in T], control_points)

    start_time = time.time()
    optimized_control_points, iter, avg_error = gradient_descent(control_points, T, X, max_iter=500)
    end_time = time.time()
    print(f"Optimization time: {end_time - start_time:.2f} seconds")
    optimized_curve = eval_bezier_curve(optimized_control_points, np.linspace(0.0, 1.0, 50))
    footpoints_of_Pc = [eval_bezier_curve(optimized_control_points, t) for t in T]

    plt.title(f"Bezier curve after gradient descent optimization \n degree= ${len(optimized_control_points)-1}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points)

    visualize_error_convergence(iter, avg_error)
    """

    #################################################
    #                                               #
    #              NACA 0012 Airfoil                #
    #                                               #
    #################################################

    # INITIALISATION

    X = generate_naca0012airfoil(100)
    #visualize_points(X)
    control_points = np.linspace(0,1,10)[:, np.newaxis]
    control_points = np.hstack((control_points, np.zeros_like(control_points)))
    tk_initial_guess = np.linspace(0.0, 1.0, len(X))

    #unoptimized_curve = eval_bezier_curve(control_points, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(control_points)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(control_points, t) for t in tk_initial_guess], control_points)

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, control_points, initial_guesses=tk_initial_guess)

    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(control_points)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(control_points, t) for t in T], control_points)

    start_time = time.time()
    optimized_control_points, iter, avg_error = gradient_descent(control_points, T, X, max_iter=300)
    end_time = time.time()
    print(f"Optimization time: {end_time - start_time:.2f} seconds")

    optimized_curve = eval_bezier_curve(optimized_control_points, np.linspace(0.0, 1.0, 50))
    footpoints_of_Pc = [eval_bezier_curve(optimized_control_points, t) for t in T]
    plt.title(f"Bezier curve after gradient descent optimization \n degree= ${len(optimized_control_points)-1}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(iter, avg_error)