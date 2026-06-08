# Fitting B-Spline

A robust and efficient implementation for fitting B-Spline [curves / surfaces] to discrete data points using [Least-Squares Optimization / Interpolation / Regression]. This project provides tools to convert noisy or scattered spatial data points into smooth, parametric B-Spline representations.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Project structure](#project-structure)
- [Usage Examples](#usage-examples)

## Installation

### Prerequisites
Make sure you have Python 3.x installed. 

### Dependencies
Install the required packages/libraries:
```bash
# Example if using Python
pip install numpy matplotlib pytest hypothesis
```

### Setup
Clone the repository to your local machine:
```bash
git clone [https://github.com/Dylan-Cmt/fitting_B-Spline.git](https://github.com/Dylan-Cmt/fitting_B-Spline.git)
cd fitting_B-Spline
```

## Features

- **Parametric Curve Generation:** Easily create, manipulate, and evaluate both Bézier and B-spline curves.
- **Custom Dataset Support:** Seamlessly integrate, manage, and use your own point cloud datasets for fitting.
- **Extensible Visualization:** Build and customize your own visualizers.
- **Robust Curve Fitting:** Fit highly accurate Bézier or B-spline curves to noisy and scattered point clouds using optimized algorithms.

## Project Structure

```text
fitting_B-Spline/
├── src/
│   ├── __init__.py
│   └── optimizers/
│       ├── bezier_curves.py        # Core Bézier representation (basis functions, evaluation)
│       ├── bezier_optimizer.py     # Fitting algorithms (PDM, TDM, SDM)
│       ├── bspline_curves.py       # Core B-Spline representation (basis functions, evaluation)
│       ├── bspline_optimizer.py    # Fitting algorithms (PDM, TDM, SDM)
│       ├── cloud_points.py         # Cloud points to fit with (you can create your owns)
│       └── visualizers.py          # All vizualizers you can use (you can create your owns)
├── tests/
│   └── tests_optimizer/
├── cloud_points_reader.py          # Function to pull cloud points from a .txt file
├── requirements.txt                # Package dependencies (numpy, scipy, matplotlib)
└── README.md                       # Project overview and usage guidelines
```

## Usage Examples

Below is a quick start example demonstrating how to run the B-Spline fitting sequence:

```python
from src.bezier_curves import *
from src.cloud_points import *
from src.visualizers import *

# 1. Generate data points from the cloud_points file (you can also read a .txt file and convert it with the cloud_points_reader.py file)
X = generate_sinus_cloud_points(50, 0) # 50 points without noise
# X = np.asarray(cloud_points_reader("your_file.txt"))

# 2. Initialize the curve with its data points (and more if it's a B-Spline)
Pc = np.array([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.0, 0.0]])

# 3. Search foot points of the initial curve before starting the optimization
T = all_tk(X, Pc)

# 3bis. Plot the initial curve with(out) its foot points if you want to make sure it is well computed
unoptimized_curve = eval_bezier_curve(Pc, np.linspace(0.0, 1.0, 50))
visualize_data_curve_controlpoints(X, unoptimized_curve, Pc)
plt.title(f"Initial Bezier curve with foot points before optimization \n degree= ${len(Pc)-1}$ and {len(X)}$ data points")
visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [eval_bezier_curve(Pc,t) for t in T], Pc)
    
# 4. Start optimization
start_time = time.time()
optimized_control_points, tot_iter, avg_error = gradient_descent_PD(Pc, T, X, max_iter=100)
end_time = time.time()
tot_time = end_time - start_time
print(f"Optimization time for PDM: {tot_time:.2f} seconds")

# 5. Plot the optimized curve
optimized_curve = eval_bezier_curve(optimized_control_points, np.linspace(0.0, 1.0, 100))
plt.title(f"Optimized curve achieved in ${tot_time:5f}$s with ${tot_iter[-1]}$ iterations\n degree= ${len(optimized_control_points)-1}$ and ${len(X)}$ data points")
visualize_data_curve_controlpoints(X, optimized_curve,optimized_control_points)

# 6. Plot the log mean error
plt.title(f"Log mean error after ${tot_iter[-1]}$ iterations \n degree= ${len(optimized_control_points)-1}$ and ${len(X)}$ data points")
visualize_error_convergence(tot_iter, avg_error)
```
