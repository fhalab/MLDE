"""
This module contains loss functions for use in calculation of the eval_metric
"""
# Import required modules
import numpy as np

# Define mse loss
def mse(real, predicted):
    """
    Calculates mean squared error.
    
    Parameters
    ----------
    real: 1D numpy array
        The true values
    predicted: 1D numpy array
        The predicted values
        
    Returns
    -------
    mse: float
        Mean squared error
    """
    # Calculate the mse
    N = len(real)
    mse = (1/N) * np.sum((real - predicted)**2)
    return mse