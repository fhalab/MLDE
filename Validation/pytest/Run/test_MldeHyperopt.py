"""
This file contains tests for all of the functions given in MldeHyperopt.
"""
# Load in necessary modules
import warnings
import numpy as np
import pandas as pd
from itertools import chain, permutations
from hyperopt import STATUS_OK, STATUS_FAIL

# Load in MLDE information
from Support.Params.Defaults import (cpu_models, gpu_models, default_model_params,
                                     default_training_params)
from Support.Params.SearchSpaces import categorical_params, space_by_model, integer_params
from Support.RunMlde.MldeHyperopt import (handle_linear_exceptions, 
                                          space_to_model_params,
                                          process_best, process_trials,
                                          optimize, run_hyperopt)

# Write a test for handle_linear_exceptions()
def test_handle_linear_exceptions():
    """
    We need to make sure that the correct changes are being made to hyperparameters
    for LinearSVR and Linear models. This function tests for...
    1. If 'dual' is set to True for linear svr, then the loss is set to 
        "squared_epsilon_insensitive". Otherwise, there should be no 'loss' 
        parameter in the function.
    2. All model parameters should be wiped for the Linear model.
    3. No other models should experience a change to their dictionaries.
    """
    # Generate model parameter dictionaries for testing
    dual_true = {"test": np.random.rand(), "dual": True}
    dual_false = {"test": np.random.rand(), "dual": False}
    linear_check = {"test": np.random.rand()}
    
    # Create a number of dictionaries for all other model types
    other_model_checks = [{"test": np.random.rand()} for _ in range(22)]
    
    # Make sure the appropriate changes are made by the handle_linear_exceptions()
    # function
    dual_true_test = handle_linear_exceptions(dual_true,
                                              "sklearn-regressor",
                                              "LinearSVR")
    assert "loss" not in dual_true_test
    assert np.array_equal(dual_true_test["test"], dual_true["test"])
    
    dual_false_test = handle_linear_exceptions(dual_false,
                                              "sklearn-regressor",
                                              "LinearSVR")
    assert "loss" in dual_false_test
    assert np.array_equal(dual_false_test["test"], dual_false["test"])
    
    linear_test = handle_linear_exceptions(linear_check,
                                           "sklearn-regressor",
                                           "Linear")
    assert linear_test == {}
    
    # Make sure no changes are made for other models
    for i, (major_model, minor_model) in enumerate(chain(cpu_models, gpu_models)):
        
        # Skip LinearSVR
        if minor_model == "LinearSVR":
            continue
                
        # Pass in test data
        test_data = other_model_checks[i]
        updated_data = handle_linear_exceptions(test_data, major_model, minor_model)
            
        if major_model == "sklearn-regressor" and minor_model == "Linear":
            assert {} == updated_data
        else:
            assert np.array_equal(test_data["test"], updated_data["test"])
            
def test_space_to_model_params_process_best():
    """
    This function tests the ability of space_to_model_params as well as 
    process_best to appropriately complete the parameter values generated
    from our search spaces. Tests include...
    1. This function just wraps "process_val()" and one additional step for adding
        input_shape to the Keras parameters. process_val has already been validated
        in a separate test script (test_CompleteKerasParams). This test will
        thus make sure that an input_shape is correctly appended for Keras models
        but nothing else.
    2. The other step is "handle_linear_exceptions" which was tested in the previous
        function. No other tests on this will be performed.
    """      
    # Define two test x_shapes. One 3d and another 2d
    x2d = (384, 1000)
    x3d = (384, 5, 200)
            
    # Now define the expected parameter values based on the Keras defaults
    expectations = {"dropout": 0.2,
                    "size1": 250,
                    "size2": 63,
                    "filter_choice": 3,
                    "n_filters1": 13,
                    "flatten_choice": "Average",
                    "filter_arch": (3, 3),
                    "n_filters2": 2}
    
    # Define a new dictionary that takes a categorical param index to a value
    integer_to_param = {}
    for param_name, param_iterable in categorical_params.items():
        integer_to_param[param_name] = {}
        for i, param_opt in enumerate(param_iterable):
            integer_to_param[param_name][i] = param_opt
    
    # Loop over all default model parameters
    for major_model, minor_model in chain(gpu_models, cpu_models):
        
        # Define x-shapes
        if minor_model in {"OneConv", "TwoConv"}:
            x_shape = x3d
        else:
            x_shape = x2d
        
        ############### Tests for space_to_model_params ########################
        # Define the expected parameters
        expected_params = {name: np.random.uniform(low=0, high=10) for name in 
                           space_by_model[major_model][minor_model]}
        
        # If this is a linear model (from sklearn), then make a blank dict as
        # the expected params
        if major_model == "sklearn-regressor" and minor_model == "Linear":
            expected_params = {}
        
        # Add a parameter for "dual" if this is a LinearSVR model
        if minor_model == "LinearSVR":
            expected_params["dual"] = True
            
        # Update Keras parameters to be correct
        input_params = expected_params.copy()       
        if major_model == "Keras":
            
            # Add input shape
            expected_params["input_shape"] = x_shape[1:]
            
            for key, val in expected_params.items():
                if key in expectations:
                    
                    # Add to the expected_params dict
                    expected_params[key] = expectations[key]
                    
                    # If this is a size parameter, then divide expectation by 1000
                    if "size" in key:
                        input_params[key] = expectations[key]/1000
                        
                    # If this is a n_filters, then divide expectation by 200
                    elif "n_filters" in key:
                        input_params[key] = expectations[key]/200
                        
                    # If this is filter_choice, then set the value to be 0.5
                    elif key == "filter_choice":
                        input_params[key] = 0.5
                        
                    # If this is filter_arch, then set the value to be (0.5, 0.5)
                    elif key == "filter_arch":
                        input_params[key] = (0.5, 0.5)
                        
                    # Otherwise, just add to input_params with no modification
                    else:
                        input_params[key] = expectations[key]
                    
        # Update the dictionary to round integer parameters down
        updated_expected_params = {}
        for key, val in expected_params.items():
            if key in integer_params:
                updated_expected_params[key] = np.floor(val)
            else:
                updated_expected_params[key] = val
                
        # Define inputs for function to be tested
        space_names = list(input_params.keys())
        space_vals = [input_params[name] for name in space_names]
                
        # Feed the function
        test_formatted_params = space_to_model_params(space_vals, space_names,
                                                      major_model, minor_model,
                                                      x_shape)
        
        # Test outputs
        assert test_formatted_params == updated_expected_params
            
        ###################### Tests for process_best ##########################
        # Identify all categorical variables
        categorical_names = [name for name in space_names if 
                             name in categorical_params]

        # If there are no categorical variables, pass default parameters directly
        # into the process_best function and make sure we meet expectations
        if len(categorical_names) == 0:
            test_best_formatted = process_best(input_params,
                                               major_model, minor_model,
                                               x_shape)
            assert test_best_formatted == updated_expected_params
            
        # If there are categorical variables, then create a new input dictionary
        # and new expected dictionary to match
        else:
            
            # Get all permutations of the category
            all_permutations = permutations(categorical_names)
            
            # Loop over permutations
            for cat_name_set in all_permutations:
                
                # Build a new input and output dictionary
                input_dict = input_params.copy()
                new_expectation = updated_expected_params.copy()
            
                # Update the new dicitonaries with categorical indices
                for categorical_name in categorical_names:
                    
                    # Get the number of options
                    n_opts = len(integer_to_param[categorical_name])
                    
                    # Loop over the options
                    for opt in range(n_opts):
                        
                        # Build a new input dictionary and a new expected dictionary
                        input_dict[categorical_name] = opt
                        new_expectation[categorical_name] = integer_to_param[categorical_name][opt]
                        
                        # Update expectations if this is LinearSVR and we are passing
                        # in dual = False
                        if categorical_name == "dual" and opt == 1:
                            assert not integer_to_param[categorical_name][opt] 
                            new_expectation["loss"] = "squared_epsilon_insensitive"
                                                        
                        # If this is filter_choice, then set the expectation value
                        # to be 5x the input value rounded up
                        elif categorical_name == "filter_choice":
                            new_expectation[categorical_name] = np.ceil(new_expectation[categorical_name] * 5)
                            
                        # If this is filter_arch, then set the expectation value
                        # to be a tuple of 5x the input value rounded up
                        elif categorical_name == "filter_arch":
                            new_expectation[categorical_name] = (np.ceil(new_expectation[categorical_name][0] * 5),
                                                                 np.ceil(new_expectation[categorical_name][1] * 5))
                        
                # Make sure that the input and new expectation are not equal
                assert new_expectation != input_dict
                
                # Run process_best and compare
                test_best_formatted = process_best(input_dict, major_model,
                                                minor_model, x_shape)
                assert test_best_formatted == new_expectation
            
def test_process_trials():
    """
    This function will test the process_trials function from MldeHyperopt. There 
    is not an easy way to deterministically generate a trials object (that's just
    the nature of hyperopt....) so we test to make sure that it is parsed correctly
    using a list of dicitonaries as a stand in object. This function is thus not testing
    the underlying assumptions of the process_trials() function, but instead to
    make sure that we are doing everything correctly within the bounds of those 
    assumptions. It is not the best test as a result. Test will be to make sure...
    1. The order in the list of dictionaries is maintained through parsing
    2. Errors in the dict are appropriately handled.
    """
    # Build the input dictionary 
    row1 = {"misc": {"vals": {"TestVar1": (0.1,), "TestVar2": (0.2,)}},
            "result": 
                {"train_err": 0.3,
                 "loss": 0.5,
                 "train_time": 1231,
                 "status": STATUS_OK}
    }
    row2 = {"misc": {"vals": {"TestVar1": (1,), "TestVar2": (23,)}},
            "result": 
                {"train_err": 0.12313,
                 "loss": 203,
                 "train_time": 14,
                 "status": STATUS_OK}
    }
    row3 = {"misc": {"vals": {"TestVar1": (15,), "TestVar2": (12.32,)}},
            "result": 
                {"status": STATUS_FAIL,
                 "message": "Something went wrong"}
    }
    row4 = {"misc": {"vals": {"TestVar1": (124,), "TestVar2": (-12.32,)}},
            "result": 
                {"train_err": 109,
                 "loss": 231,
                 "train_time": 988,
                 "status": STATUS_OK}
    }
    test_trial_info = [row1, row2, row3, row4]
    
    # Build the expected output
    expected_array = (("TestMajor", "TestMinor", 0, 1231, 0.3, 0.5, "TestVar1", 0.1),
                      ("TestMajor", "TestMinor", 0, 1231, 0.3, 0.5, "TestVar2", 0.2),
                      ("TestMajor", "TestMinor", 1, 14, 0.12313, 203, "TestVar1", 1),
                      ("TestMajor", "TestMinor", 1, 14, 0.12313, 203, "TestVar2", 23),
                      ("TestMajor", "TestMinor", 2, 0, "Something went wrong", "Something went wrong", "TestVar1", 15),
                      ("TestMajor", "TestMinor", 2, 0, "Something went wrong", "Something went wrong", "TestVar2", 12.32),
                      ("TestMajor", "TestMinor", 3, 988, 109, 231, "TestVar1", 124),
                      ("TestMajor", "TestMinor", 3, 988, 109, 231, "TestVar2", -12.32))
    expected_df = pd.DataFrame(expected_array, 
                               columns = ("MajorModel", "SpecificModel", "HyperRound",
                                          "RunTime", "TrainErr", "TestErr", "Hyper", "HyperVal"))
    
    # Run process trials and compare to expectation
    trial_output = process_trials(test_trial_info, "TestMajor", "TestMinor")
    assert trial_output.equals(expected_df)
    