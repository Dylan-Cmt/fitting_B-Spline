import matplotlib.pyplot as plt

#Visualize a point on the curve at parameter tk
def visualize_point(point, tk):
    plt.scatter(point[0], point[1], color='red', label=f'Point on curve at t={tk:.2f}')

# Visualize the generated cloud points
def visualize_points(points):
    x, y = zip(*points)  # Unzip the list of points into x and y coordinates
    plt.scatter(x, y, color='black', alpha=0.5, s=5)
    # title is set in main.py
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # Set equal aspect ratio
    plt.grid()
    plt.show()

# Visualize 2 sets of points (e.g., cloud points and curve points)
def visualize_two_sets_of_points(points1, points2):
    x1, y1 = zip(*points1)  # Unzip the first set of points into x and y coordinates
    x2, y2 = zip(*points2)
    plt.scatter(x1, y1, color='black', alpha=0.5, s=5, label='Cloud Points')
    plt.scatter(x2, y2, color='red', alpha=0.5, s=5, label='Curve Points')
    # title is set in main.py
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')  # Set equal aspect ratio
    plt.grid()
    plt.legend()
    plt.show()