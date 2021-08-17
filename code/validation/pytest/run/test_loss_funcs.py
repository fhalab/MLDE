"""
This module tests the loss functions found in MLDE.Support.RunMlde.LossFuncs.py
"""
# Import necessary functions
from ....run_mlde.loss_funcs import mse
import numpy as np
from sklearn.metrics import mean_squared_error

# Define the test function
def test_mse():
    """
    What this function confirms:
    1) The user-defined mean-squared error function matches the sklearn mean-
    squared error function
    2) The user-defined mse function is symmetric
    3) THe user-defined mse function correctly calculates mse
    """
    # Build a random array of true and predicted values
    real_vals = np.random.rand(10000)
    predicted_vals = np.random.rand(10000)
    
    # Caculate mse with sklearn and the user-defined metric
    user_def_result = mse(real_vals, predicted_vals)
    sklearn_result = mean_squared_error(real_vals, predicted_vals)
    assert np.isclose(user_def_result, sklearn_result)
    
    # MSE should be symmetric. Make sure
    assert user_def_result == mse(predicted_vals, real_vals)
    
    # Now make an array for which you know the expected mse
    real_vals = np.array([0, 1, 2, 3, 4])
    predicted_vals = np.array([2, 2, 2, 4, 0])
    assert mse(real_vals, predicted_vals) == 4.4
    