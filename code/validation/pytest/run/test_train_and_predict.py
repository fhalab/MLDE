"""
This file contains a function which checks the results and effectiveness of
train_and_predict() from MLDE.Support.RunFuncs.TraininAndPredict()
"""
# Load necessary modules
from ....params.defaults import (GPU_MODELS, CPU_MODELS, DEFAULT_MODEL_PARAMS,
                                 DEFAULT_TRAINING_PARAMS)
from ....run_mlde.mlde_classes import MldeModel
from ....run_mlde.complete_keras_params import process_val
from ....run_mlde.train_and_predict import train_and_predict

from itertools import chain
import numpy as np
from sklearn.model_selection import KFold
import pytest

# Filter convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Write a function to complete keras params
def temporary_complete_keras(major_model, specific_model, 
                             model_params, test_x):
    """
    This function is designed only to complete Keras parameters. It wraps 
    process_val() to make sure that the non-integer parameters are correctly
    translated prior to being used to build keras models
    """
    # If a keras model, add to model params
    if major_model == "Keras":
        if specific_model in {"OneConv", "TwoConv"}:
            model_params["input_shape"] = test_x.shape[1:]
            model_params = {key: process_val(key, val, test_x.shape)
                            for key, val in model_params.items()}
        else:
            final_x_shape = np.prod(test_x.shape[1:])
            model_params["input_shape"] = (final_x_shape,)
            model_params = {key: process_val(key, val, (len(test_x), final_x_shape))
                            for key, val in model_params.items()}
            
    return model_params

# Write a function that instantiates all mlde models
def instantiate_all_mlde(full_test_x):
    
    # Instantiate a number of models
    all_models = []
    for major_model, specific_model in chain(GPU_MODELS, CPU_MODELS):
        
        # Build model params
        model_params = DEFAULT_MODEL_PARAMS[major_model][specific_model].copy()
        model_params = temporary_complete_keras(major_model, specific_model,
                                                model_params, full_test_x)
        
        # Instantiate the model
        all_models.append(MldeModel(major_model, specific_model,
                                    model_params = model_params,
                                    training_params = DEFAULT_TRAINING_PARAMS[major_model].copy()))

    return all_models
def test_train_and_predict():
    """
    Things this function confirms:
    - Both sampled_x and x_to_predict are appropriately reshaped when passed 
    in to train_and_predict with the _reshape_x flag thrown, but not otherwise
    - We throw an exception when an error is raised
    - The outputs of an exception are what we expect them to be
    - Passing a list into train_and_predict n_cv results in a list being passed
    in to MldeModel.train_cv()
    - The returned predictions match the expected shape based on the value of
    x passed in
    """
    # Build a test x and y
    full_test_x = np.random.rand(1000, 5, 10)
    training_x = full_test_x[:100]
    training_y = np.random.rand(100)
    full_test_x_copy = full_test_x.copy()
    training_x_copy = training_x.copy()
    training_y_copy = training_y.copy()
    
    # Build a bad train x
    bad_train_x = training_x.copy().astype(object)
    bad_train_x[10] = "ADFADFADF"
    
    # Decide on the expected shapes
    expected_2D_full = (1000, 50)
    expected_2D_train = (100, 50)
    
    # Instantiate all models
    all_models = instantiate_all_mlde(full_test_x)
    
    # Create a set of train-test inds
    splitter = KFold(n_splits = 5)
    split_inds = list(splitter.split(training_x))
    
    # Run everything with reshape set to true
    for model in all_models:
        
        # Run the function and get the data
        (training_loss, testing_loss,
         mean_preds, stdev_preds,
         sampled_x, x_to_predict,
         train_test_inds) = train_and_predict(model,
                                              sampled_x = training_x, 
                                              sampled_y = training_y,
                                              x_to_predict = full_test_x,
                                              train_test_inds = split_inds,
                                              _reshape_x = True,
                                              _debug = True)
        
        # Make sure our sampled_x and x_to_predict are the correct shape
        if model.specific_model in {"OneConv", "TwoConv"}:
            assert sampled_x.shape == training_x.shape
            assert x_to_predict.shape == full_test_x.shape
        else:
            assert sampled_x.shape == expected_2D_train
            assert x_to_predict.shape == expected_2D_full
            
            # Make sure that reshaping x didn't screw anything up
            for i, subx in enumerate(sampled_x):
                assert np.array_equal(subx, training_x[i].flatten())
            for i, subx in enumerate(x_to_predict):
                assert np.array_equal(subx, full_test_x[i].flatten())
                
        # Make sure the loss arrays are floats
        assert len(training_loss) == 5
        assert len(testing_loss) == 5
        assert len(training_loss.shape) == 1
        assert len(testing_loss.shape) == 1
        assert not np.array_equal(training_loss, testing_loss)
        
        # Make sure the prediction arrays are the correct shape
        assert len(mean_preds) == len(x_to_predict)
        assert len(stdev_preds) == len(x_to_predict)
        assert len(mean_preds.shape) == 1
        assert len(stdev_preds.shape) == 1
        
        # Make sure that n_cv is a list and that we recover split_inds
        assert isinstance(train_test_inds, list)
        for i, (train_inds, test_inds) in enumerate(split_inds):
            assert np.array_equal(train_inds, train_test_inds[i][0])
            assert np.array_equal(test_inds, train_test_inds[i][1])
            
        # Now attempt to run train_and_predict in a manner that we know will
        # throw a warning
        with pytest.warns(UserWarning, match = "Error when training .+"):
            
            # Run the function and get the data
            (fake_train, fake_test,
             fake_preds, fake_stdevs) = train_and_predict(model,
                                                           sampled_x = bad_train_x, 
                                                           sampled_y = training_y,
                                                           x_to_predict = full_test_x,
                                                           train_test_inds = split_inds,
                                                           _reshape_x = True)
        
        # Make sure the fake output is what we expect
        assert fake_train == np.inf
        assert fake_test == np.inf
        assert np.all(fake_preds == 0)
        assert np.all(fake_stdevs == 0)
        
        # Make sure our training information has not changed
        assert np.array_equal(full_test_x_copy, full_test_x)
        assert np.array_equal(training_x_copy, training_x)
        assert np.array_equal(training_y_copy, training_y)