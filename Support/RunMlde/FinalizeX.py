"""
This function in this file reshapes x values as appropriate for the different
inbuilt model classes. The OneConv and TwoConv submodels in KerasModels expect
a 2D input, while all other models expect a 1D input. This function reshapes
x as appropriate to meet this requirement.
"""
# Import necessary modules
import numpy as np

# Write a function that reformats x as appropriate
def finalize_x(major_model, specific_model, x):
    """
    Reshapes x values as appropriate for the different inbuilt model classes.
    The OneConv and TwoConv submodels in KerasModels expect a 2D input, while
    all other models expect a 1D input. This function reshapes x as appropriate 
    to meet this requirement.
    
    Parameters
    ----------
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
    x: 3D numpy array
        The embedding matrix to reshape
        
    Returns
    -------
    reshaped x: 2D or 3D numpy array
        The input is returned for Keras OneConv and Keras TwoConv. Otherwise, 
        the last two dimensions are flattened, returning a 2D array.
    """
    # If the shape of x is not 3D, throw an error
    if len(x.shape) != 3:
        raise ValueError("Input X must be 3D")
    
    # If this is a convolutional keras model, do nothing
    if major_model == "Keras" and specific_model in {"OneConv", "TwoConv"}:
        return x
    
    # Otherwise, flatten x along the amino acid dimensions
    flat_length = np.prod(x.shape[1:])
    return np.reshape(x, [len(x), flat_length]) 