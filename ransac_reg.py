import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def linear_ransac_curve_fit(x, y):
    x1 = np.array(x).reshape((-1, 1))
    y1 = np.array(y).reshape((-1, 1))
    xi = np.linspace(min(x), max(x), 500).reshape((-1, 1))
    
    reg = linear_model.RANSACRegressor(linear_model.LinearRegression())
    reg.fit(x1, y1)
    yi = reg.predict(xi)
    coeff = reg.estimator_.coef_
    intercept = reg.estimator_.intercept_[0]
    coeff = np.array([intercept, coeff[0, 0]])

    inliers = reg.inlier_mask_
    outliers = np.logical_not(inliers)

    plt.plot(x[inliers], y[inliers], 'k.', label='inliers')
    plt.plot(x[outliers], y[outliers], 'r.', label='outliers')
    plt.plot(xi, yi, label='Linear Regression')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear')
    print('Equation: {0:.5f} + {1:.5f}x'.format(coeff[0], coeff[1]))
    print('Y-intercept: {}'.format(coeff[0]))
    plt.legend()


def quadratic_ransac_curve_fit(x, y, name):
    x1 = x.reshape((-1, 1))
    y1 = y.reshape((-1, 1))

    xi = np.linspace(min(x), max(x), 500).reshape((-1, 1))

    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x1)
    xi_2 = poly_2.fit_transform(xi)

    reg = linear_model.RANSACRegressor(linear_model.LinearRegression())
    reg.fit(x_2, y1)
    yi = reg.predict(xi_2)
    coeff = reg.estimator_.coef_
    intercept = reg.estimator_.intercept_[0]
    coeff = np.array([intercept, coeff[0, 1], coeff[0, 2]])

    inliers = reg.inlier_mask_
    outliers = np.logical_not(inliers)

    plt.plot(x[inliers], y[inliers], 'k.', label='inliers')
    plt.plot(x[outliers], y[outliers], 'r.', label='outliers')
    plt.plot(xi, yi, label='Quadratic Curve')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic')
    print('Equation: {0:.5f} + {1:.5f}x + {2:.5f}x^2'.format(coeff[0], coeff[1], coeff[2]))
    print('Y-intercept: {}'.format(coeff[0]))
    plt.legend()
    
def cubic_ransac_curve_fit(x, y):
    x1 = x.reshape((-1, 1))
    y1 = y.reshape((-1, 1))

    xi = np.linspace(min(x), max(x), 500).reshape((-1, 1))

    poly_3 = PolynomialFeatures(degree=3)
    x_3 = poly_3.fit_transform(x1)
    xi_3 = poly_3.fit_transform(xi)

    reg = linear_model.RANSACRegressor(linear_model.LinearRegression())
    reg.fit(x_3, y1)
    yi = reg.predict(xi_3)
    coeff = reg.estimator_.coef_
    intercept = reg.estimator_.intercept_[0]
    coeff = np.array([intercept, coeff[0, 1], coeff[0, 2], coeff[0, 3]])

    inliers = reg.inlier_mask_
    outliers = np.logical_not(inliers)

    plt.plot(x[inliers], y[inliers], 'k.', label='inliers')
    plt.plot(x[outliers], y[outliers], 'r.', label='outliers')
    plt.plot(xi, yi, label='Cubic Curve')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic')
    print('Equation: {0:.5f} + {1:.5f}x + {2:.5f}x^2 + {3:.5f}x^3'.format(coeff[0], coeff[1], coeff[2], coeff[3]))
    print('Y-intercept: {}'.format(coeff[0]))
    plt.legend()
