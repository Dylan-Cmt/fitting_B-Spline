import random as rd
import numpy as np

# Generate random circle shaped cloud points with noise with radius 1 and center at (0, 0)
def generate_circle_cloud_points(num_points, noise_level):
    points = []
    rd.seed(2025)
    for _ in range(num_points):
        angle = rd.uniform(0, 2 * np.pi)
        radius = 1 + rd.uniform(-noise_level, noise_level)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))
    return points

# Generate random points along the edges of a square (side length 2, centered at (0, 0)) with noise
def generate_square_cloud_points(num_points, noise_level):
    points = []
    rd.seed(2026)
    for i in range(num_points):
        # forcing datapoints to begin and end at the expremitis of my initial curve
        if i == 0:
            points.append((0, 0))
            continue       
        if i == num_points - 1:
            points.append((1, 0))
            continue
        # Randomly choose an edge (0: bottom, 1: right, 2: top, 3: left)
        edge = rd.randint(1, 3) # [0,3] for entire square, [1,3] for only 3 edges (no bottom edge)
        if edge == 0:  # Bottom edge
            x = rd.uniform(0, 1) + rd.uniform(-noise_level, noise_level)
            y = rd.uniform(-noise_level, noise_level)
        elif edge == 1:  # Right edge
            x = 1 + rd.uniform(-noise_level, noise_level)
            y = rd.uniform(0, 1) + rd.uniform(-noise_level, noise_level)
        elif edge == 2:  # Top edge
            x = rd.uniform(0, 1) + rd.uniform(-noise_level, noise_level)
            y = 1 + rd.uniform(-noise_level, noise_level)
        else:  # Left edge
            x = rd.uniform(-noise_level, noise_level)
            y = rd.uniform(0, 1) + rd.uniform(-noise_level, noise_level)
        points.append((x, y))
    return points

# Generate random C shaped cloud points with noise with radius 1 and center at (0, 0)
def generate_c_cloud_points(num_points, noise_level):
    points = []
    rd.seed(2025)
    while len(points) < num_points:
        angle = rd.uniform(0, 2 * np.pi)
        
        if np.pi / 4 < angle < 7 * np.pi / 4:
            radius = 1 + rd.uniform(-noise_level, noise_level)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append((x, y))
    
    return np.array(points)

# Generate random points on a sinus curve
def generate_sinus_cloud_points(num_points, noise_level):
    points = []
    rd.seed(2025)
    for _ in range(num_points):
        x = rd.uniform(0, 1)
        y = np.sin(x*np.pi) + rd.uniform(-noise_level, noise_level)
        points.append((x, y))
    return np.array(points)