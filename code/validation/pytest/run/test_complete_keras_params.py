"""
This file contains functions needed to test the effectiveness of the process_val
function. 
"""
# Load necessary modules
from itertools import chain
import pytest

def test_process_val():
    """
    Things this function confirms:
    1) If the correct shape is passed in to process_val, then the correct changes
    are made to Keras models
    2) Only keras model parameters are affected by process_val
    3) The appropriate built-in flags are thrown if the x_shape isn't what we
    expect for a given parameter name
    """
    # Start by loading the default model parameters
    from ....params.defaults import DEFAULT_MODEL_PARAMS, CPU_MODELS, GPU_MODELS
    from ....run_mlde.complete_keras_params import process_val
        
    # Define two test x_shapes. One 3d and another 2d
    x2d = (384, 1000)
    x3d = (384, 5, 200)
    
    # Define the expected number of latent dims for each input shape
    expected_latent_dims_2d = 1000
    expected_latent_dims_3d = 200
        
    # Now define the expected parameter values based on the Keras defaults
    expectations = {"dropout": 0.2,
                    "size1": 250,
                    "size2": 63,
                    "filter_choice": 3,
                    "n_filters1": 13,
                    "flatten_choice": "Average",
                    "filter_arch": (3, 3),
                    "n_filters2": 2}
            
    # Loop over the different models and test the effects of process_val
    for major_model, specific_model in chain(GPU_MODELS, CPU_MODELS):
        
        # Pull the default model params
        test_params = DEFAULT_MODEL_PARAMS[major_model][specific_model].copy()
        
        # Determine the appropriate shape
        if major_model == "Keras" and specific_model in {"OneConv", "TwoConv"}:
            test_shape = x3d
        else:
            test_shape = x2d
        
        # Run the parameters through process_val
        processed_params = {key: process_val(key, val, test_shape) for 
                            key, val in test_params.items()}
        
        # Assert that the parameters match what we expect
        if major_model == "Keras":
            for key, val in processed_params.items():
                assert expectations[key] == val
        else:
            assert all(test_params[key] == val for 
                       key, val in processed_params.items())
            
    # Make sure we are calculating the correct number of latent dimensions
    returned_latent_dims = process_val(key, val, x2d, _debug = True)
    assert returned_latent_dims == expected_latent_dims_2d
    
    returned_n_aas, returned_latent_dims = process_val(key, val,
                                                       x3d, _debug = True)
    assert returned_latent_dims == expected_latent_dims_3d
    assert returned_n_aas == 5
    
    # Add assertions that will catch expected failures
    with pytest.raises(ValueError, match="Input X must be 2 or 3D"):
        process_val(key, val, (2, 3, 4, 2))
        
    with pytest.raises(AssertionError, match="Expect a 3D array for convolutional networks"):
        process_val("filter_choice", 0.25, (2, 4))
    
    with pytest.raises(AssertionError, match="Expect a 3D array for convolutional networks"):
        process_val("filter_arch", 0.25, (2, 4))
        
    with pytest.raises(AssertionError, match="Expect a 2D array for feed forward networks"):
        process_val("size1", 0.25, (2, 4, 6))