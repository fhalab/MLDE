"""
This file contains the function needed for completing model_params for Keras
models during hyperparameter optimization. The Keras parameters are defined as
floats between 0 and 1 in the search space, but need to be converted to integers.
The floats represent either "percent of latent dimensions" (e.g. 0.25 would 
become 100 for a 400-latent-dimension embedding) or "percent of number of amino
acids" (e.g. 0.25 would become 1 for a 4-amino-acid combinatorial space).

The function defined here also converts floats to integers for non-Keras models
when the model is expecting an integer value but hyperopt outputs a float.
"""
# Import numpy
import numpy as np

# Import mlde objects needed
from Support.Params.SearchSpaces import latent_perc_params, integer_params

# Define a function that processes float model params for Keras functions and 
# converts them to integers
def process_val(name, value, x_shape, _debug = False):
    """
    Converts floats to integers when floats are output by hyperopt but integers
    are needed for model training. Also converts percentile values in Keras 
    search spaces to absolute integers.
    
    Parameters
    ----------
    name: str
        Name of the variable. 
    value: float or int
        The value associated with the variable name
    x_shape: tuple
        Gives the shape of the input x-values for a model. This is used to
        calculate the appropriate conversions from percentile to integer for
        keras parameters
        
    Returns
    -------
    value: int
        The updated value
    """
    # If this is a 3D array, extract both n_aas and n_latent_dims. Otherwise,
    # just get n_latent_dims
    n_x_dims = len(x_shape)
    if n_x_dims == 3:
        n_aas = x_shape[1]
        n_latent_dims = x_shape[2]
        if _debug:
            return n_aas, n_latent_dims
    elif n_x_dims == 2:
        n_latent_dims = x_shape[1]
        if _debug:
            return n_latent_dims
    else:
        raise ValueError("Input X must be 2 or 3D")
    
    # If this is a filter param, calculate the filter width
    if name == "filter_choice":
        assert n_x_dims == 3, "Expect a 3D array for convolutional networks"
        value = int(np.ceil(value * n_aas))
    elif name == "filter_arch":
        assert n_x_dims == 3, "Expect a 3D array for convolutional networks"
        value = tuple([int(np.ceil(subval * n_aas)) for subval in value])
    elif name == "size1":
        assert n_x_dims == 2, "Expect a 2D array for feed forward networks"
        
    # For keras parameters where the search space is defined by the fraction of
    # total latent space (e.g. the number of nodes for fully connected networks)
    if name in latent_perc_params:
        value = np.ceil(value * n_latent_dims)
        
    # Convert to an integer if we are in the "to_integer" set
    if name in integer_params:
        value = int(value)
    
    # Return the value. It may be unmodified if none of the above conditions
    # were met
    return value
