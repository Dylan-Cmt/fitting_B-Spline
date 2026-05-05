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
    
    
    # INITIALISATION

    #X = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])
    X = generate_sinus_cloud_points(50, 0.02)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.4, 0.0],
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"simple example - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"simple example - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)


    #################################################
    #                                               #
    #                 Bézier curve                  #
    #                                               #
    #################################################

    
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Bézier curve - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Bézier curve - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    

    #################################################
    #                                               #
    #                   C shape                     #
    #                                               #
    #################################################

    
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"C shape - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"C shape - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    


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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Square shape - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Square shape - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    """


    #################################################
    #                                               #
    #              NACA 0012 Airfoil                #
    #                                               #
    #################################################

    
    # INITIALISATION

    X = generate_naca0012airfoil(100)
    #visualize_points(X)
    Pc = np.array([
        [0.0, 0.0],
        [1/4, 0.0],
        [2/4, 0.0],
        [3/4, 0.0],
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"NACA 0012 Airfoil - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"NACA 0012 Airfoil - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    


    #################################################
    #                                               #
    #            Curve 1 thèse Claire               #
    #                                               #
    #################################################

    
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Curve 1 thèse Claire - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Curve 1 thèse Claire - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    

    #################################################
    #                                               #
    #            Curve 2 thèse Claire               #
    #                                               #
    #################################################

    
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Curve 2 thèse Claire - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Curve 2 thèse Claire - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    

    #################################################
    #                                               #
    #       Circle shape with closed Bspline        #
    #                                               #
    #################################################
    
    
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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Circle shape with closed Bspline - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Circle shape with closed Bspline - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
    

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
#    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    
    # OPTIMIZATION

    # first foot points calculus
    T = all_tk(X, Pc, knots, degree)
    footpoints_on_the_curve = np.array([
        bspline_curve(Pc, t, knots, degree) for t in T
    ])

    #plt.title(f"Bspline curve with foot points before optimization \n degree= ${degree}$ and ${len(X)}$ data points")
#    visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, footpoints_on_the_curve, Pc)
    
    start_time1 = time.time()
    optimized_control_points1, nb_iter1, avg_error1 = gradient_descent_PD(Pc, T,knots, degree, X, max_iter=1000)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")
    tot_iter1 = 10**(nb_iter1[-1])

    optimized_curve1 = np.array([bspline_curve(optimized_control_points1, t, knots, degree) for t in t_vals])
    footpoints_of_Pc = np.array([
        bspline_curve(optimized_control_points1, t, knots, degree) for t in T
    ])
    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1} iterations")
    #visualize_data_curve_controlpoints(X, optimized_curve1, optimized_control_points1)

    # convergence of the average error
    #visualize_error_convergence(nb_iter1, avg_error1)

    start_time2 = time.time()
    optimized_control_points2, nb_iter2, avg_error2 = gradient_descent_TD(Pc, T,knots, degree, X, max_iter=1000)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    optimized_curve2 = np.array([bspline_curve(optimized_control_points2, t, knots, degree) for t in t_vals])

    plt.title(f"Double Ellipsoid - PDM VS TDM (PDM {tot_time1:.2f}s, TDM {tot_time2:.2f}s) with {int(tot_iter1)} iterations")
    #sym:visualize_data_curve1_curve2
    visualize_data_curve1_curve2(X, optimized_curve1, optimized_curve2)
    plt.title(f"Double Ellipsoid - Convergence PDM VS TDM")
    #sym:visualize_two_error_convergence
    visualize_two_error_convergence(nb_iter1, avg_error1, avg_error2)
