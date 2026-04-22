import matplotlib.pyplot as plt

#Visualize a point on the curve at parameter tk
def visualize_point(point, tk):
    plt.scatter(point[0], point[1], label=f'Point on curve at t={tk:.2f}')

# Visualize the generated cloud points
def visualize_points(points):
    x, y = zip(*points)  # Unzip the list of points into x and y coordinates
    plt.scatter(x, y, color='black', alpha=0.5, s=5)
    # title is set in main.py
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # Set equal aspect ratio
    plt.grid()
    plt.legend()
    plt.show()

def visualize_control_points(control_points):
    x, y = zip(*control_points)
    plt.scatter(x, y, color='green', s=50, marker='x', label='Control Points')

# Visualize 2 sets of points (e.g., cloud points and curve points)
def visualize_two_sets_of_points(cloud, curve):
    x1, y1 = zip(*cloud)  # Unzip the first set of points into x and y coordinates
    x2, y2 = zip(*curve)
    plt.scatter(x1, y1, color='black', alpha=0.5, s=5, label='Cloud Points')
    plt.plot(x2, y2, color='red', linewidth=1, label='Curve')
    #plt.scatter(x2, y2, color='red', alpha=0.5, s=5, label='Curve Points')
    # title is set in main.py
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # Set equal aspect ratio
    plt.grid()
    plt.legend()
    plt.show()

def visualize_data_curve_footpoints_controlpoints(cloud, curve, foot_points, control_points):
    x1, y1 = zip(*cloud)  # Unzip the first set of points into x and y coordinates
    x2, y2 = zip(*curve)
    x3, y3 = zip(*foot_points)
    x4, y4 = zip(*control_points)
    plt.scatter(x1, y1, color='green', alpha=0.5, s=30, label='Cloud Points')
    plt.scatter(x4, y4, color='red', s=50, marker='x', label='Control Points')
    plt.plot(x2, y2, color='red', linewidth=1, label='Curve')
    plt.scatter(x3, y3, color='blue', alpha=0.5, s=10, label='Foot Points')
    # title is set in main.py
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # Set equal aspect ratio
    plt.grid()
    plt.legend()
    plt.show()

def visualize_error_convergence(log_iter, log_avg_error):
    plt.plot(log_iter, log_avg_error, 'o-')
    plt.xlabel('Log10(Iteration)')
    plt.ylabel('Log10(Average Error)')
    plt.title('Convergence of Gradient Descent')
    plt.show()