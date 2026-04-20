from src.bezier_curves import *
from src.bezier_optimizer import *
from src.cloud_points import *
from src.visualizers import *
import time

if __name__ == "__main__":
    
    ##############################################################################
    #                                                                            #
    #                         Cloud points generation                            #
    #                                                                            #
    ##############################################################################
    
    """
    num_points = 100  
    noise_level = 0.0
    # circle or square or c or sinus
    cloud_points = generate_sinus_cloud_points(num_points, noise_level)
    plt.title('Sinus Shaped Cloud Points with Noise Level 0.0')
    visualize_points(cloud_points)
    """

    ##############################################################################
    #                                                                            #
    #                         B-Spline curve initialization                      #
    #                                                                            #
    ##############################################################################
    
    """
    num_points = 100  
    noise_level = 0.0
    # circle or square or c or sinus
    cloud_points = generate_sinus_cloud_points(num_points, noise_level)

    knots = np.concatenate([
    np.zeros(3),
    np.linspace(0, 1, 8),
    np.ones(3)
    ])

    degree = 3
    
    
    control_points = np.linspace(0,1,10)[:, np.newaxis]
    control_points = np.hstack((control_points, np.zeros_like(control_points)))
    
    control_points = np.array(
    [
        [0,   0],
        [1/9, 0],
        [2/9, 0],
        [3/9, 0],
        [4/9, 0],
        [5/9, 0],
        [6/9, 0],
        [7/9, 0],
        [8/9, 0],
        [9/9, 0],
    ]
    )
    

    curve_points = evalbsplinecurve(degree, knots, control_points, sample=50)

    plt.title('Sinus Shaped Cloud Points without Noise and Initial B-Spline Curve')
    visualize_two_sets_of_points(cloud_points, curve_points)
    """
    
    ##############################################################################
    #                                                                            #
    #               Visualization of the curve + a specific point                #
    #                                                                            #
    ##############################################################################

    """num_points = 100
    noise_level = 0.0
    # circle or square or c or sinus
    cloud_points = generate_sinus_cloud_points(num_points, noise_level)

    knots = np.concatenate([
    np.zeros(3),
    np.linspace(0, 1, 8),
    np.ones(3)
    ])

    degree = 3
    
    control_points = np.array(
    [
        [0,   0],
        [1/9, 0],
        [2/9, 0],
        [3/9, 0],
        [4/9, 0],
        [5/9, 0],
        [6/9, 0],
        [7/9, 0],
        [8/9, 0],
        [9/9, 0],
    ]
    )

    tk = 32/100
    point_on_curve_at_tk = evalbsplinecurve_at_t(degree, knots, control_points, tk)
    a_random_point = np.array([0.5, 1.2]) # could be a data point
    curve_points = evalbsplinecurve(degree, knots, control_points, sample=50)
    # plot tk on the curve
    plt.title('B-Spline Curve with a Specific Point at t=0.32 and a Random Point')
    visualize_point(point_on_curve_at_tk, tk)
    visualize_point(a_random_point, 222222) # second argument can be random too
    visualize_points(curve_points)
    """
    
    ##############################################################################
    #                                                                            #
    #               Visualization of the curve + its derivative                  #
    #                                                                            #
    ##############################################################################  
    
    """
    knots = np.concatenate([
    np.zeros(3),
    np.linspace(0, 1, 8),
    np.ones(3)
    ])

    degree = 3
    
    control_points = np.array(
    [
        [0,   0],
        [1/9, 0],
        [2/9, 0],
        [3/9, 0],
        [4/9, 1],
        [5/9, 0],
        [6/9, 0],
        [7/9, 0],
        [8/9, 0],
        [9/9, 0],
    ]
    )


    curve_points = evalbsplinecurve(degree, knots, control_points, sample=50)
    curve_points_prime = evalbsplinecurve_derivative(degree, knots, control_points, sample=300)
    visualize_two_sets_of_points(curve_points, curve_points_prime)
    """

    ##############################################################################
    #                                                                            #
    #               Visualization of data + curve + foot points                  #
    #                                                                            #
    ##############################################################################  

    """
    cloud_points = np.array([[0.0, 0.0],[0.5, 0.5], [1.0, 0.0]])

    knots = np.array([0,0,0,1,1,1])
    degree = 2
    
    control_points = np.array(
    [
        [0  ,   0],
        [0.4,   0],
        [1.0,   0]
    ])

    initial_guess = [0.0, 0.1, 1.0]
    curve_points = evalbsplinecurve(degree, knots, control_points, sample=50)
    
    t = []
    for i in range(len(cloud_points)):
        tk = newton_tk(cloud_points[i], degree, knots, control_points, initial_guess[i], tol=1e-6, max_iter=100)
        t.append(tk)
    Pt = [evalbsplinecurve_at_t(degree, knots, control_points, tk) for tk in t]
    #visualize_control_points(control_points)
    visualize_data_curve_footpoints(cloud_points, curve_points, Pt)
    """
    

    ##############################################################################
    #                                                                            #
    #                                test                                        #
    #                                                                            #
    ##############################################################################  

    """
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    control_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    tk_initial_guess = [0.0, 0.1, 1.0]
    """

    ""    
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
    ""

    """
    X = generate_sinus_cloud_points(100, 0.1)
    err_per_nb_ctrl_pts = []
    pts_max = 7
    for nb_pts_ctrl in range(3, pts_max+1):
        control_points = np.linspace(0,1,nb_pts_ctrl)[:, np.newaxis]
        control_points = np.hstack((control_points, np.zeros_like(control_points)))
        tk_initial_guess = np.linspace(0.0, 1.0, len(X))
        
        T = all_tk(X, control_points, initial_guesses=tk_initial_guess)


        optimized_control_points, iter, avg_error = gradient_descent(control_points, T, X, max_iter=500)
        optimized_curve = eval_bezier_curve(optimized_control_points, np.linspace(0.0, 1.0, 50))
        footpoints_of_Pc = [eval_bezier_curve(optimized_control_points, t) for t in T]
        err_per_nb_ctrl_pts.append(avg_error[-1])
    
    plt.title("Final average error vs number of control points to approximate a sinusoidal curve with 100 noisy data points and 500 iterations")
    x = range(2, pts_max)
    y = err_per_nb_ctrl_pts
    plt.plot(x, y, marker='o')
    plt.xlabel("Degree")
    plt.ylabel("Final average error")
    plt.xticks(x)
    plt.grid()
    plt.show()
    """