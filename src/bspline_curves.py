import numpy as np

def bspline_basis(i, degree, t):
    n = len(knots) - 1
    if degree == 0:
        if i >= n:
            return 0.0
        return 1.0 if knots[i] <= t < knots[i + 1] or (t == knots[-1] and knots[i + 1] == knots[-1]) else 0.0
    first_part = 0.0
    second_part = 0.0
    if (i + degree) < n:
        denom1 = knots[i + degree] - knots[i]
        if denom1 != 0:
            first_part = (t - knots[i]) / denom1 * bspline_basis(i, degree - 1, t)
    if (i + degree + 1) < n:
        denom2 = knots[i + degree + 1] - knots[i + 1]
        if denom2 != 0:
            second_part = (
                (knots[i + degree + 1] - t)
                / denom2
                * bspline_basis(i + 1, degree - 1, t)
            )
    return first_part + second_part
 
def dt_bspline_basis(i, degree, t): # cf p59 du NURBS BOOK
    first_part, second_part = 0.0, 0.0
    denom1 = knots[i + degree] - knots[i]
    if denom1 != 0:
        first_part  = degree / denom1 * bspline_basis(i, degree - 1, t)
    denom2 = knots[i + degree + 1] - knots[i + 1]
    if denom2 != 0:
        second_part = degree / denom2 * bspline_basis(i + 1, degree - 1, t)
    return first_part - second_part

def dtt_bspline_basis(i, degree, t): # cf p62 du NURBS BOOK
    first_part, second_part, third_part = 0.0, 0.0, 0.0

    # N_{i, p-2}
    denom1 = (knots[i + degree] - knots[i]) * (knots[i + degree - 1] - knots[i])
    if denom1 != 0:
        first_part = degree * (degree - 1) / denom1 * bspline_basis(i, degree - 2, t)

    # N_{i+1, p-2}
    denom2a = (knots[i + degree] - knots[i]) * (knots[i + degree] - knots[i + 1])
    denom2b = (knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree] - knots[i + 1])

    coeff2 = 0.0
    if denom2a != 0:
        coeff2 += 1.0 / denom2a
    if denom2b != 0:
        coeff2 += 1.0 / denom2b

    second_part = degree * (degree - 1) * coeff2 * bspline_basis(i + 1, degree - 2, t)

    # N_{i+2, p-2}
    denom3 = (knots[i + degree + 1] - knots[i + 1]) * (knots[i + degree + 1] - knots[i + 2])
    if denom3 != 0:
        third_part = degree * (degree - 1) / denom3 * bspline_basis(i + 2, degree - 2, t)

    return first_part - second_part + third_part

 
def bspline_curve(P, t):
    t_min = knots[degree]
    t_max = knots[-degree - 1]
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = bspline_basis(i, degree, t)
        curve += N * P[i]
    return curve
 
def dt_bspline_curve(P, t):
    t_min = knots[degree]
    t_max = knots[-degree - 1]
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = dt_bspline_basis(i, degree, t)
        curve += N * P[i]
    return curve

def dtt_bspline_curve(P, t):
    t_min = knots[degree]
    t_max = knots[-degree - 1]
    curve = np.zeros(P.shape[1])
    for i in range(len(P)):
        N = dtt_bspline_basis(i, degree, t)
        curve += N * P[i]
    return curve