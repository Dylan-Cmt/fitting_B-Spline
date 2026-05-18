from cloud_points_reader import *
import numpy as np
from src.bspline_curves import *
from src.bspline_optimizers import *
from src.cloud_points import *
from src.visualizers import *
import time

def compare3methods(P, T, X, max_iter=100, constraint=True):
    # PDM
    start_time_PD = time.time()
    optimized_control_points_PD, tot_iter_PD, avg_error_PD = gradient_descent_PD(P, T, knots, degree, X, max_iter=max_iter, constraint=constraint)
    end_time_PD = time.time()
    tot_time_PD = end_time_PD - start_time_PD
    print(f"Optimization time for PDM: {tot_time_PD:.2f} seconds")

    # TDM
    start_time_TD = time.time()
    optimized_control_points_TD, tot_iter_TD, avg_error_TD = gradient_descent_TD(P, T, knots, degree, X, max_iter=max_iter, constraint=constraint)
    end_time_TD = time.time()
    tot_time_TD = end_time_TD - start_time_TD
    print(f"Optimization time for TDM: {tot_time_TD:.2f} seconds")

    # SDM
    start_time_SD = time.time()
    optimized_control_points_SD, tot_iter_SD, avg_error_SD = gradient_descent_SD(P, T, knots, degree, X, max_iter=max_iter, constraint=constraint)
    end_time_SD = time.time()
    tot_time_SD = end_time_SD - start_time_SD
    print(f"Optimization time for SDM: {tot_time_SD:.2f} seconds")

    # curves
    curve_PD = np.array([bspline_curve(optimized_control_points_PD, t, knots, degree) for t in np.linspace(0, 1, 100)])
    curve_TD = np.array([bspline_curve(optimized_control_points_TD, t, knots, degree) for t in np.linspace(0, 1, 100)])
    curve_SD = np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in np.linspace(0, 1, 100)])

    # Comparison of results
    plt.title(f"PDM VS TDM VS SDM with ${round(10**(tot_iter_PD[-1]))}$ iterations \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve1_curve2_curve3(X, curve_PD, curve_TD, curve_SD)

    plt.title(f"Convergence PDM VS TDM VS SDM")
    visualize_three_error_convergence(tot_iter_PD, avg_error_PD, avg_error_TD, avg_error_SD)

    return {
        "control_points": [optimized_control_points_PD, optimized_control_points_TD, optimized_control_points_SD],
        "curves": [curve_PD, curve_TD, curve_SD],
        "errors": [avg_error_PD, avg_error_TD, avg_error_SD],
        "iterations": [tot_iter_PD, tot_iter_TD, tot_iter_SD]
    }

if __name__ == "__main__":

    #################################################
    #                                               #
    #               simple example                  #
    #                                               #
    #################################################
    
    """
    # INITIALISATION

    #X = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])
    X = generate_sinus_cloud_points(50, 0.02)
    #visualize_points(X)
    Pc = np.array([[0.0, 0.0],[0.1, 0.0],[0.4, 0.0],[1.0, 0.0]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Sinus shape \n")
    results = compare3methods(Pc,T,X, max_iter=10)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    Pc = np.linspace(0,1,4)[:, np.newaxis]
    Pc = np.hstack((Pc, np.zeros_like(Pc)))
    degree = 3
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 10)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Bézier curve \n")
    results = compare3methods(Pc,T,X, max_iter=50)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate C shape \n")
    results = compare3methods(Pc,T,X, max_iter=50)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
    """

    #################################################
    #                                               #
    #                 Square shape                  #
    #                                               #
    #################################################
    
    """
    # INITIALISATION

    X = generate_square_cloud_points(200, 0)
    #visualize_points(X)
    Pc = np.array([ [0.0, 0.0], [0.0, 0.8],[0.20, 1.0],[0.8, 1.0], [1.0, 0.8], [1.0, 0.0]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Square shape \n")
    results = compare3methods(Pc,T,X, max_iter=100)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    #Pc = np.linspace(0,1,5)[:, np.newaxis] ; Pc = np.hstack((Pc, np.zeros_like(Pc)))
    Pc = np.array([[0.0, 0.0],[0.0, 0.05],[0.7, 0.05],[1.0, 0.0]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate NACA \n")
    results = compare3methods(Pc,T,X, max_iter=10)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    #Pc = np.array([[0.0, 0.0], [1/3, 2/15], [2/3, 4/15], [1.0, 0.4]])
    Pc = np.array([[0.0, 0.0], [0.51, 0.29], [0.76, 0.38], [1.0, 0.4]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Curve 1 \n")
    results = compare3methods(Pc,T,X, max_iter=10)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    #Pc = np.array([[-0.1, 0.0], [0, 1/30], [0.1, 2/30], [0.2, 0.1]])
    Pc = np.array([[-0.1, 0.0],[-0.1, 0.05],[-0.075, 0.075],[0.0, 0.1],[0.2, 0.1]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Curve 2 \n")
    results = compare3methods(Pc,T,X, max_iter=100)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
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
    Pc = np.array([[0.0, -1.0],[-1.0, -1.0],[-1.0,  1.0],[1.0,  1.0],[1.0, -1.0],[0.0, -1.0]])
    degree = 2
    n = len(Pc)
    # knot size = nb_ctrlpts + degree + 1
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Circle shape \n")
    results = compare3methods(Pc,T,X, max_iter=50)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
    """

    #################################################
    #                                               #
    #                Double Ellipsoid               #
    #                                               #
    #################################################
    
    """
    # INITIALISATION
    X = cloud_points_reader("double_ellipsoid.txt")
    #visualize_points(X)
    
    Pc = np.array([[-0.03, 0.013],[-0.025, 0.025],[0.0, 0.025],[0.01, 0.025],[0.016, 0.02],[0.016, -0.012],[0.013, -0.015],[0.0, -0.015],[-0.10, -0.002],[-0.03, 0.013]])
    

    degree = 2
    # knot size = nb_ctrlpts + degree + 1
    n = len(Pc)
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
    tk_initial_guess = np.linspace(0.0, 1.0, len(X))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Double Ellipsoid \n")
    results = compare3methods(Pc,T,X, max_iter=50)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
    """

    #################################################
    #                                               #
    #                      Apollo                   #
    #                                               #
    #################################################

    """
    X = cloud_points_reader("apollo.txt")
    #visualize_points(X)

    Pc = np.array([[13, 65],[17.5,76],[25,65],[115,8],[130,0],[115,-8],[28,-65],[15,-75],[12,-65],[-10,0.0],[13, 65]])

    degree = 3
    # knot size = nb_ctrlpts + degree + 1
    n = len(Pc)
    knots = np.arange(n - degree + 1) / (n -degree)
    knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))

    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    t_vals = np.linspace(0, 1, 100)
    unoptimized_curve = np.array([bspline_curve(Pc, t, knots, degree) for t in t_vals])
    #plt.title(f"Initial Bspline curve before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    print("Simulate Apollo \n")
    results = compare3methods(Pc,T,X, max_iter=10, constraint=True)

    #optimized_control_points_SD = results["control_points"][2]
    #visualize_data_curve_controlpoints(X, np.array([bspline_curve(optimized_control_points_SD, t, knots, degree) for t in t_vals]), optimized_control_points_SD)
    """