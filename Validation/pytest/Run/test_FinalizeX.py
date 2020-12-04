"""
This module tests to be sure that finalize_x() works.
"""
# Load in the appropriate parameters
from Support.Params.Defaults import cpu_models, gpu_models
from Support.RunMlde.FinalizeX import finalize_x
from itertools import chain
import pytest
import numpy as np

# Write a function for testing finalize x
def test_finalize_x():
    """
    What this function confirms:
    1) finalize_x correctly reshapes the data to be a 3D or 2D matrix depending
    on the model type 
    2) Reshaping x to flatten the last dimensions does not change any ordering
    3) An error is thrown if we pass in a 2D array
    """
    # Make a 2D x. finalize_x should fail on this.
    bad_x_input = np.array([10, 15])
    
    # Make a good 3D x
    good_x_input = np.random.rand(1000, 10, 25)
    
    # Define expected output shapes
    expected_output_shape_3d = good_x_input.shape
    expected_output_shape_2d = (good_x_input.shape[0], 
                                good_x_input.shape[1] * good_x_input.shape[2])
    
    # Loop over all cpu and gpu models
    for major_model, specific_model in chain(cpu_models, gpu_models):
        
        # Make sure we fail on the bad x    
        with pytest.raises(ValueError, match="Input X must be 3D"):
            finalize_x(major_model, specific_model, bad_x_input)
        
        # Run finalize_x
        output_x = finalize_x(major_model, specific_model, good_x_input)
            
        # If this is keras, x should have the same shape as the input
        if major_model == "Keras" and specific_model in {"OneConv", "TwoConv"}:
            assert np.array_equal(good_x_input, output_x)
            assert output_x.shape == expected_output_shape_3d
            
        # Otherwise, it should have the later dimensions flattened
        else:
            assert output_x.shape == expected_output_shape_2d
            for i, row in enumerate(good_x_input):
                assert np.array_equal(row.flatten(), output_x[i])