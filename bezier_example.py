from cloud_points_reader import *
from src.bezier_curves import *
from src.bezier_optimizer import *
from src.cloud_points import *
from src.visualizers import *
import time

def compare3methods(P,T,X, max_iter=100):
    start_time1 = time.time()
    optimized_control_points1, tot_iter1, avg_error1 = gradient_descent_PD(P, T, X, max_iter=max_iter)
    end_time1 = time.time()
    tot_time1 = end_time1 - start_time1
    print(f"Optimization time for PDM: {tot_time1:.2f} seconds")

    start_time2 = time.time()
    optimized_control_points2, tot_iter2, avg_error2 = gradient_descent_TD(P, T, X, max_iter=max_iter)
    end_time2 = time.time()
    tot_time2 = end_time2 - start_time2
    print(f"Optimization time for TDM: {tot_time2:.2f} seconds")

    start_time3 = time.time()
    optimized_control_points3, tot_iter3, avg_error3 = gradient_descent_SD(P, T, X, max_iter=max_iter)
    end_time3 = time.time()
    tot_time3 = end_time3 - start_time3
    print(f"Optimization time for SDM: {tot_time2:.2f} seconds")

    plt.title(f"PDM VS TDM with ${round(10**(tot_iter1[-1]))}$ iterations \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    visualize_data_curve1_curve2_curve3(X, eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100)), eval_bezier_curve(optimized_control_points2, np.linspace(0.0, 1.0, 100)), eval_bezier_curve(optimized_control_points3, np.linspace(0.0, 1.0, 100)))
    plt.title(f"Convergence PDM VS TDM VS SDM")
    visualize_three_error_convergence(tot_iter1, avg_error1, avg_error2, avg_error3)


if __name__ == "__main__":
    
    #################################################
    #                                               #
    #               simple example                  #
    #                                               #
    #################################################

    # INITIALISATION

    """
    X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    Pc = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    tk_initial_guess = [0.0, 0.1, 1.0]
    """
    
    """
    #X = np.array([[0.0, 0.0], [0.4, 0.6], [0.6, 0.4], [1.0, 0.0]])
    X = generate_sinus_cloud_points(50, 0)
    #X = generate_square_cloud_points(100, 0.04)
    #Pc = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    #Pc = np.array([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.0, 0.0]])
    Pc = np.linspace(0,1,3)[:, np.newaxis]
    Pc = np.hstack((Pc, np.zeros_like(Pc)))    
    
    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X,10)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
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
    Pc = np.linspace(0,1,10)[:, np.newaxis]
    Pc = np.hstack((Pc, np.zeros_like(Pc)))

    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 10)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """

    #################################################
    #                                               #
    #                 Bézier curve                  #
    #                                               #
    #################################################

    
    # INITIALISATION
    # bezier curve to fit with
    Beziercontrol_points = np.array([[0.0, 0.0], [0.3, 0.5], [2.4, 0.4], [1.0, 0.0]])
    X = eval_bezier_curve(Beziercontrol_points, np.linspace(0.0, 1.0, 50))
    #visualize_points(X)


    Pc = np.linspace(0,1,10)[:, np.newaxis]
    Pc = np.hstack((Pc, np.zeros_like(Pc)))
    tk_initial_guess = np.linspace(0.0, 1.0, len(X))

    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 100)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    

    #################################################
    #                                               #
    #            Curve 1 thèse Claire               #
    #                                               #
    #################################################

    """
    X = generate_curve1(100)
    #visualize_points(X)

    Pc = np.array([[0.0, 0.0], [0.33, 0.133], [0.66, 0.266], [1.0, 0.4]])

    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 10)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """

    #################################################
    #                                               #
    #            Curve 2 thèse Claire               #
    #                                               #
    #################################################

    """
    X = generate_curve2(100)
    #visualize_points(X)

    Pc = np.array([[-0.1, 0.0], [0.0, 1/30], [0.1, 2/30], [0.2, 0.1]])
    # Pc = np.array([[-0.1, 0.0], [-1/40, 1/40], [2/40, 2/40], [5/40, 3/40], [0.2, 0.1]])
    
    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 10)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """

    #################################################
    #                                               #
    #                   C shape                     #
    #                                               #
    #################################################
    
    """
    # INITIALISATION
    X = generate_c_cloud_points(100,0)
    #visualize_points(X)

    Pc = np.linspace(-0.5, 0.5, 10)[:, np.newaxis]
    Pc = np.hstack((np.ones_like(Pc), Pc))
    
    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 100)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """

    #################################################
    #                                               #
    #                      Apollo                   #
    #                                               #
    #################################################

    """
    X = cloud_points_reader("apollo.txt")
    #visualize_points(X)

    #Pc = np.linspace(0,125,11)[:, np.newaxis]
    #Pc = np.hstack((Pc, np.zeros_like(Pc)))
    #Pc = np.vstack((Pc, [0.0, 0.0]))

    #Pc = np.array([ [0, 0], [0, 120], [250, 0], [0, -120], [0, 0] ])

    Pc = np.array([  [0, 0], [0, 60], [0, 120],[125, 80], [250, 0],[125, -80], [0, -120],[0, -60], [0, 0] ])

    #Pc = np.array([[0, 0], [0, 40],[0, 80],[0, 120], [83, 80],[166, 40],[250, 0], [166, -40],[83, -80],[0, -120], [0, -80],[0, -40],[0, 0],[0, 0],[0, 0] ])

    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 100)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """

    #################################################
    #                                               #
    #             Double ellipsoidale               #
    #                                               #
    #################################################

    
    """Pc = np.linspace(-0.06,0.0175,10)[:, np.newaxis]
    Pc = np.hstack((Pc, np.zeros_like(Pc)))
    Pc = np.vstack((Pc, [0.0, 0.0]))"""
    
    """
    X = np.asarray(cloud_points_reader("double_ellipsoid.txt"))
    #visualize_points(X)
    n = 10
    N = len(X)
    indices = np.round(np.linspace(0, N-1, n-1)).astype(int)
    Pc = np.vstack([X[indices], X[0]])
    #visualize_control_points(Pc)
    
    # OPTIMIZATION

    T = all_tk(X, Pc)
    
    #unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
    #plt.title(f"Initial Bézier curve and foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
    #plt.title(f"Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc, t) for t in T], Pc)
    
    compare3methods(Pc,T,X, 100)

    #optimized_curve = eval_bezier_curve(optimized_control_points1, np.linspace(0.0, 1.0, 100))
    #footpoints_of_Pc = [eval_bezier_curve(optimized_control_points1, t) for t in T]

    #plt.title(f"Optimized curve achieved in ${tot_time1:5f}$s with ${tot_iter1[-1]}$ iterations \n degree= ${len(optimized_control_points1)-1}$ and ${len(X)}$ data points")
    #visualize_data_curve_footpoints_controlpoints(X, optimized_curve, footpoints_of_Pc, optimized_control_points1)

    #visualize_error_convergence(tot_iter1, avg_error1)
    """