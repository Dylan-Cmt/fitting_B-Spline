from cloud_points_reader import *
import numpy as np
from src.bspline_curves import *
from src.bspline_optimizers import *
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
    # INITIALISATION

    X = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])
    #X = generate_sinus_cloud_points(50, 0)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 0.0]
    ])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    #visualize_error_convergence(nb_iter, avg_error)
    """


    #################################################
    #                                               #
    #                 Bézier curve                  #
    #                                               #
    #################################################

    """
    # INITIALISATION

    X = generate_bezier_curve(100)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [1/3, 0.0],
        [2/3, 0.0],
        [1.0, 0.0]
    ])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    #visualize_error_convergence(nb_iter, avg_error)
    """

    #################################################
    #                                               #
    #                   C shape                     #
    #                                               #
    #################################################

    """
    # INITIALISATION

    X = generate_c_cloud_points(100)
    #visualize_points(X)
    Pc = np.linspace(-0.5, 0.5, 6)[:, np.newaxis]
    Pc = np.hstack((np.ones_like(Pc), Pc))
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    
    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """


    #################################################
    #                                               #
    #                 Square shape                  #
    #                                               #
    #################################################
    
    """
    # INITIALISATION

    X = generate_square_cloud_points(1000, 0)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [-0.5, 1.5],
        [0.5, 1.0],
        [1.5, 1.5],
        [1.0, 0.0]
    ])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """


    #################################################
    #                                               #
    #              NACA 0012 Airfoil                #
    #                                               #
    #################################################

    """
    # INITIALISATION

    X = generate_naca0012airfoil(100)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [1/3, 0.0],
        [2/3, 0.0],
        [1.0, 0.0]
    ])
    degree = 3
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=500)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """


    #################################################
    #                                               #
    #            Curve 1 thèse Claire               #
    #                                               #
    #################################################

    """
    # INITIALISATION

    X = generate_curve1(100)
    #visualize_points(X)
    Pc = np.array([[0.0, 0.0], [0.33, 0.133], [0.66, 0.266], [1.0, 0.4]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """

    #################################################
    #                                               #
    #            Curve 2 thèse Claire               #
    #                                               #
    #################################################

    """
    # INITIALISATION

    X = generate_curve2(100)
    #visualize_points(X)
    Pc = np.array([[-0.1, 0.0], [0, 1/30], [0.1, 2/30], [0.2, 0.1]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """

    #################################################
    #                                               #
    #       Circle shape with closed Bspline        #
    #                                               #
    #################################################
    
    """
    # INITIALISATION

    X = generate_circle_cloud_points(100, 0)
    #visualize_points(X)
    Pc = np.array([
        [  0.0, -1.0],
        [ -1.0, -1.0],
        [ -1.0,  1.0],
        [  1.0,  1.0],
        [  1.0, -1.0],
        [  0.0, -1.0],
    ])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    
    t_vals  = np.linspace(0, 1, 300)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    """

    #################################################
    #                                               #
    #                Double Ellipsoid               #
    #                                               #
    #################################################
    
    
    # INITIALISATION
    X = cloud_points_reader("double_ellipsoid.txt")
    #visualize_points(X)
    
    Pc = np.array([
        [ 0.016,  -0.015],
        [ -0.10,   0.000],
        [ -0.03,   0.013],
        [-0.025,   0.025],
        [ 0.016,   0.025],
        [ 0.016,   0.005],
        [ 0.016,  -0.015]
    ])
    degree = 3
    # knot size = nb_ctrlpts + degree + 1
    n = len(Pc)
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    tk_initial_guess = np.linspace(0.0, 1.0, len(X))

    t_vals  = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time = time.time()
    optimized_control_points, nb_iter, avg_error = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=100)
    end_time = time.time()
    tot_time = end_time - start_time
    tot_iter = 10**(nb_iter[-1])

    optimized_curve = np.array([bspline_curve(optimized_control_points, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points, t, knots, degree) for t in T
    ])
    plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter}$ iterations")
    visualize_data_curve_controlpoints(X, optimized_curve, optimized_control_points)

    # convergence of the average error
    visualize_error_convergence(nb_iter, avg_error)
    