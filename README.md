# Fitting B-Spline

A robust and efficient implementation for fitting curves to discrete data points using PDM/TDM/SDM.

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

## Usage Example

Below is a quick start example demonstrating how to run the Bézier fitting sequence:

```python
from src.optimizers.curves import *
from src.optimizers.bezier_optimizer import *
#from src.optimizers.bspline_optimizers import *
from src.optimizers.cloud_points import *
from src.optimizers.visualizers import *
import time

# 1. Generate data points from the cloud_points file (you can also read a .txt file and convert it with the cloud_points_reader.py file)
X = generate_sinus_cloud_points(50, 0) # 50 points without noise
# X = np.asarray(cloud_points_reader("your_file.txt"))

# 2. Initialize the curve with its control points (and more if it's a B-Spline)
Pc = np.array([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.0, 0.0]])
Bz0 = BezierCurve(Pc)

# 3. Search foot points of the initial curve before starting the optimization
T = all_tk(X, Bz0.P)

# 3bis. Plot the initial curve with(out) its foot points if you want to make sure it is well computed
t_vals = np.linspace(0.0, 1.0, 100)
unoptimized_curve = Bz0.eval(t_vals)
visualize_data_curve_controlpoints(X, unoptimized_curve, Bz0.P)
plt.title(
    f"Initial Bézier curve with foot points\n"
    f"degree = {Bz0.degree}, {len(X)} data points"
)
visualize_data_curve_footpoints_controlpoints(X, unoptimized_curve, [Bz0.eval(t) for t in T], Bz0.P)
    
# 4. Start optimization then create a new Bézier object
start_time = time.time()
optimized_control_points, tot_iter, log_avg_error, log_max_error = gradient_descent_PD(Bz0.P, T, X, max_iter=75)
end_time = time.time()
tot_time = end_time - start_time
print(f"Optimization time for PDM: {tot_time:.2f} seconds")
Bz1 = BezierCurve(optimized_control_points)

# 5. Plot the optimized curve
optimized_curve = Bz1.eval(t_vals)
plt.title(
    f"Optimized curve in {tot_time:.2f}s with {int(10**tot_iter[-1])} iterations\n"
    f"degree = {Bz1.degree}, {len(X)} data points\n"
    f"Max error = {float(10**log_max_error[-1])}"
)
visualize_data_curve_controlpoints(X, optimized_curve,Bz1.P)

# 6. Plot the log mean error
plt.title(
    f"Log max error after {int(10**tot_iter[-1])} iterations\n"
    f"degree = {Bz1.degree}, {len(X)} data points"
)
visualize_error_convergence(tot_iter, log_avg_error)
```
