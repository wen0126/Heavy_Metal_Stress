
# The function is used to verify the inversion results with two indices
import numpy as np

# Calculation of correlation R；
def compute_correlation(x,y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    ssr = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(0,len(x)):
        diff_xbar = x[i] - xbar
        dif_ybar = y[i] - ybar
        ssr += (diff_xbar * dif_ybar)
        var_x += diff_xbar**2
        var_y += dif_ybar**2
    sst = np.sqrt(var_x * var_y)
    return ssr/sst

# Calculate RMSE; Difference between observation and model inversion (root mean square error)
def compute_rmse(x, y):
    rmse = 0.0
    diff = 0.0
    var = 0.0
    for i in range(0, len(x)):
        diff = x[i] - y[i]
        var += diff ** 2
    rmse = np.sqrt(var / len(x))
    return rmse