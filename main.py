from cloud_points import *
from curves import *
from visualizers import *

if __name__ == "__main__":
    
    ##############################################################################
    #                                                                            #
    #                         Cloud points generation                            #
    #                                                                            #
    ##############################################################################
    num_points = 100  
    noise_level = 0.05
    # circle or square or c or sinus
    cloud_points = generate_sinus_cloud_points(num_points)
    #visualize_points(cloud_points)

    ##############################################################################
    #                                                                            #
    #                         B-Spline curve initialization                      #
    #                                                                            #
    ##############################################################################
    nodes = np.concatenate([
    np.zeros(3),
    np.linspace(0, 1, 8),
    np.ones(3)
    ])

    degree = 3

    control_points = np.arange(10)[:, np.newaxis]
    control_points = np.hstack((control_points, np.zeros_like(control_points)))
    """
    control_points = np.array(
    [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 1],
        [5, 0],
        [6, 0],
        [7, 0],
        [8, 0],
        [9, 0],
    ]
    )
    """

    curve_points = evalbsplinecurve(degree, nodes, control_points, sample=300)
    #visualize_points(curve_points)

    #plt.title('Sinus Shaped Cloud Points without Noise')
    #visualize_two_sets_of_points(cloud_points, curve_points)

    ##############################################################################
    #                                                                            #
    #               Visualization of the curve + a specific point                #
    #                                                                            #
    ##############################################################################

    t_min = nodes[degree]
    t_max = nodes[-degree - 1]

    tk = 28455/30000
    point_on_curve_at_tk = evalbsplinecurve_at_t(degree, nodes, control_points, tk)
    random_point = np.array([4, 2])
    # plot tk on the curve
    #visualize_point(point_on_curve_at_tk, tk)
    #visualize_point(random_point, tk)
    #visualize_points(curve_points)

    ##############################################################################
    #                                                                            #
    #               Visualization of the curve + its derivative                  #
    #                                                                            #
    ##############################################################################  

    curve_points_prime = evalbsplinecurve_derivative(degree, nodes, control_points, sample=300)
    point_on_curve_prime_at_tk = evalbsplinecurve_derivative_at_t(degree, nodes, control_points, tk)
    visualize_point(point_on_curve_prime_at_tk, tk)
    visualize_two_sets_of_points(curve_points, curve_points_prime)