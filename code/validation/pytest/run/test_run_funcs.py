"""
This file validates all functions found in the MLDE.Support.RunMlde.RunFuncs file
"""
# Import necessary modules
from importlib import reload
import numpy as np
import pandas as pd
import warnings
import pytest
import pickle
from itertools import chain
from copy import deepcopy
from sklearn.model_selection import KFold
from ....run_mlde.run_funcs import (process_results, get_training_from_design,
                                    run_mlde, prep_input_data, run_hyperopt_mlde,
                                    combine_results, run_mlde_cl)
from ....params.defaults import (CPU_MODELS, GPU_MODELS, DEFAULT_MODEL_PARAMS)
from ....params.search_spaces import SPACE_BY_MODEL

def test_process_results():
    """
    This function confirms:
    - Unprocessed results from train_and_predict are correctly unpacked
    - Predictions are correctly stacked during the stacking phase
    - All auxilliary arrays (sorted_train_loss, sorted_test_loss, sorted_preds,
    sorted_stds, and sorted_model_names) are correct results from sorting
    - The summary_df matches the ordering of the sorted train_loss, test_loss
    and model names
    - Compound predictions are accurately calculated and returned
    - Predictions are correctly mapped to combinations
    - Predictions are correctly ordered in the results_df
    - Samples that were in the training data are correctly identified
    - The correct number of models are averaged
    - Exceptions are handled well: We correctly sort models that return np.inf
    for their testing and training error
    """       
    # Generate fake input data. The input data comes directly from 
    # train_and_predict, and consists of a list of tuples of numpy arrays
    predictions = np.random.rand(4, 1000)
    stdevs = np.random.rand(4, 1000)
    synthetic_unprocessed_results = [(100.0, 23.0, predictions[0], stdevs[0]),
                                     (0.3, 12.2, predictions[1], stdevs[1]),
                                     (111.1, 0.3, predictions[2], stdevs[2]),
                                     (0.1, 12.1, predictions[3], stdevs[3])]
    
    # Generate model names
    model_names = np.array(["ADFADF", "ADFACAE", "AECADKH", "ADCKJAE"])
    
    # Create a combo to ind and a set of training inds
    combo_to_ind = {str(i): i for i in range(1000)}
    training_inds = np.random.choice(np.arange(1000), size = 100, replace = False)
    set_of_training_inds = set(training_inds)
        
    # Calculate the expected compound preds
    expected_compound_preds = np.array([predictions[2],
                                        (predictions[2] + predictions[3]) / 2,
                                        (predictions[2] + predictions[3] + predictions[1]) / 3,
                                        (predictions[2] + predictions[3] + predictions[1] + predictions [0]) / 4])
    
    # Calculate the expected results dataframe
    expected_df_preds = expected_compound_preds[1]
    expected_df_list = [[str(i), pred, "YES"] if i in set_of_training_inds else
                        [str(i), pred, "NO"] for i, pred in enumerate(expected_df_preds)]
    expected_df = pd.DataFrame(expected_df_list, columns = ("AACombo", 
                                                            "PredictedFitness",
                                                            "InTrainingData?"))
    expected_df.sort_values(by = "PredictedFitness", ascending = False,
                            inplace = True)
    
    # Set the expected summary_df
    expected_summary_df = pd.DataFrame([["AECADKH", 111.1, 0.3],
                                        ["ADCKJAE", 0.1, 12.1],
                                        ["ADFACAE", 0.3, 12.2],
                                        ["ADFADF", 100.0, 23.0]],
                                        columns = ("ModelName", 
                                                   "cvTrainingError",
                                                   "cvTestingError"))
    
    # Now package expected results 
    expected_results = {
        "all_train_loss": np.array([100.0, 0.3, 111.1, 0.1,]),
        "all_test_loss": np.array([23.0, 12.2, 0.3, 12.1]),
        "all_preds": predictions,
        "all_stds": stdevs,
        "sorted_train_loss": np.array([111.1, 0.1, 0.3, 100.0]),
        "sorted_test_loss": np.array([0.3, 12.1, 12.2, 23.0]),
        "sorted_preds": np.array([predictions[2], predictions[3], predictions[1], predictions[0]]),
        "sorted_stds": np.array([stdevs[2], stdevs[3], stdevs[1], stdevs[0]]),
        "sorted_model_names": np.array(["AECADKH", "ADCKJAE", "ADFACAE", "ADFADF"]),
        "compound_preds": expected_compound_preds,
        "results_df": expected_df,
        "summary_df": expected_summary_df
                        }
    numpy_comparisons = {"all_train_loss", "all_test_loss", "all_preds",
                         "all_stds", "sorted_train_loss", "sorted_test_loss",
                         "sorted_preds", "sorted_stds", "sorted_model_names",
                         "compound_preds"}
    
    # Run process_results
    output_order = ("all_train_loss", "all_test_loss", "all_preds", "all_stds",
                    "sorted_train_loss", "sorted_test_loss", "sorted_preds",
                    "sorted_stds", "sorted_model_names", "compound_preds", 
                    "results_df", "summary_df")
    processed_results = process_results(synthetic_unprocessed_results, 
                                        model_names, 2, combo_to_ind,
                                        training_inds, _debug = True)
    
    # Make sure the outputs of processed results are what we expect
    for test_output_name, test_output in zip(output_order, processed_results):
        
        # Check numpy results
        if test_output_name in numpy_comparisons:
            if test_output_name == "sorted_model_names":
                assert tuple(test_output) == tuple(expected_results[test_output_name])
            else:
                assert np.allclose(test_output, expected_results[test_output_name])
            
        # Check pandas results
        else:
            assert test_output.equals(expected_results[test_output_name])
            
    # Replace the last element of testing loss with np.inf. This should not 
    # change anything in the output
    synthetic_unprocessed_results = [(100.0, np.inf, predictions[0], stdevs[0]),
                                     (0.3, 12.2, predictions[1], stdevs[1]),
                                     (111.1, 0.3, predictions[2], stdevs[2]),
                                     (0.1, 12.1, predictions[3], stdevs[3])]
    
    # Change expected results slightly
        # Set the expected summary_df
    expected_summary_df = pd.DataFrame([["AECADKH", 111.1, 0.3],
                                        ["ADCKJAE", 0.1, 12.1],
                                        ["ADFACAE", 0.3, 12.2],
                                        ["ADFADF", 100.0, np.inf]],
                                        columns = ("ModelName", 
                                                   "cvTrainingError",
                                                   "cvTestingError"))
    
    # Now package expected results 
    expected_results = {
        "all_train_loss": np.array([100.0, 0.3, 111.1, 0.1,]),
        "all_test_loss": np.array([np.inf, 12.2, 0.3, 12.1]),
        "all_preds": predictions,
        "all_stds": stdevs,
        "sorted_train_loss": np.array([111.1, 0.1, 0.3, 100.0]),
        "sorted_test_loss": np.array([0.3, 12.1, 12.2, np.inf]),
        "sorted_preds": np.array([predictions[2], predictions[3], predictions[1], predictions[0]]),
        "sorted_stds": np.array([stdevs[2], stdevs[3], stdevs[1], stdevs[0]]),
        "sorted_model_names": np.array(["AECADKH", "ADCKJAE", "ADFACAE", "ADFADF"]),
        "compound_preds": expected_compound_preds,
        "results_df": expected_df,
        "summary_df": expected_summary_df
                        }
    processed_results = process_results(synthetic_unprocessed_results, 
                                        model_names, 2, combo_to_ind,
                                        training_inds, _debug = True)
    # Make sure the outputs of processed results are what we expect
    for test_output_name, test_output in zip(output_order, processed_results):
        
        # Check numpy results
        if test_output_name in numpy_comparisons:
            if test_output_name == "sorted_model_names":
                assert tuple(test_output) == tuple(expected_results[test_output_name])
            else:
                assert np.allclose(test_output, expected_results[test_output_name])
            
        # Check pandas results
        else:
            assert test_output.equals(expected_results[test_output_name])

def test_get_training_from_design():
    """
    This function confirms:
    - Training inds correctly map to combos and vice versa
    - The correct combos are pulled from the normalized_design_space to get the
    training embeddings. This can be checked using their size.
    """
    # Create a fake training data dataframe
    fake_training_data = pd.DataFrame([["1", 23.1],
                                       ["9", 20.1],
                                       ["4", 31.2]],
                                       columns = ("AACombo", "Fitness"))
    
    # Create a combo-to-ind dict
    combo_to_ind = {str(i): i for i in range(10)}
    
    # Create a design space
    design_space = np.random.rand(10, 2)
    
    # Get the expected training inds and training fitness
    expected_training_inds = np.array([1, 9, 4])
    expected_training_fitness = np.array([23.1, 20.1, 31.2])
    
    # Get the expected training embeddings
    expected_embeddings = np.array([design_space[1], design_space[9], 
                                    design_space[4]])
    
    # Run get_training_from_design
    (training_inds, 
     embedded_combos,
     training_fitness) = get_training_from_design(fake_training_data, 
                                                  combo_to_ind, design_space)
    
    # Make sure our expectations are met
    assert np.array_equal(expected_training_inds, training_inds)
    assert np.array_equal(expected_training_fitness, training_fitness)
    assert np.array_equal(embedded_combos, expected_embeddings)

def test_run_mlde():
    """
    This function confirms:
    - If training data is passed in as a tuple, then it is correctly extracted
    - If training data is not passed in as a tuple, then it is correctly constructed
    - The model names correctly map to the constructed models after all 
    processing is complete
    - The _return_processed flag caused processing and gives the expected
    results based on process_results
    - The default parameter values do not change from beginning to end
    """
    # Grab the original default parameter values
    import code.params.defaults as spd
    orig_default_mod_params = deepcopy(spd.DEFAULT_MODEL_PARAMS)
    orig_default_train_params = deepcopy(spd.DEFAULT_TRAINING_PARAMS)
    
    # Build fake input data. Make copies so that we can make comparisons later.
    full_design_space = np.random.rand(10000, 3, 100)
    full_design_space_copy = full_design_space.copy()
    combo_to_ind = {str(i): i for i in range(len(full_design_space))}
    training_inds = np.random.choice(np.arange(len(full_design_space)),
                                     size = 10, replace = False)
    training_inds_copy = training_inds.copy()
    training_fitness = np.random.rand(10)
    training_fitness_copy = training_fitness.copy()
    training_embeddings = full_design_space[training_inds]
    training_embeddings_copy = training_embeddings.copy()
    training_df = pd.DataFrame([[str(training_ind), training_fitness[i]]
                                for i, training_ind in enumerate(training_inds)],
                               columns = ("AACombo", "Fitness"))
    training_data = (training_inds, training_embeddings, training_fitness)
    
    # Import large datasets
    import code.run_mlde.run_funcs as runfcs
    
    # Define a set of cross-validation indices
    kfold_splitter = KFold(n_splits = 5, shuffle = True)
    train_test_inds = list(kfold_splitter.split(training_inds))
    
    # Identify GPU and CPU models
    parameter_df = pd.read_csv("./code/validation/basic_test_data/TestMldeParams.csv")
    
    # Instantiate gpu and cpu models. Package info for hyperparameter optimization.
    default_mods, _ = prep_input_data(parameter_df, training_embeddings.shape)
    
    # Build the expected model names
    expected_model_names = np.array([f"{major_model}-{minor_model}" for 
                                     major_model, minor_model in 
                                     chain(GPU_MODELS, CPU_MODELS)])
    
    # If we don't run _reshape_x, we expect an assertion error
    with pytest.warns(UserWarning, match = "Error when training .+"):
        no_mulitprocess_results = run_mlde(default_mods, training_data,
                                           full_design_space, combo_to_ind,
                                           train_test_inds = train_test_inds,
                                           _reshape_x = False, _debug = True)
    
    # Run the function, but package training arguments in a tuple
    no_mulitprocess_results = run_mlde(default_mods, training_data,
                                       full_design_space, combo_to_ind,
                                       train_test_inds = train_test_inds,
                                        _reshape_x = True, _debug = True)    
    
    # Make sure nothing has changed
    assert np.array_equal(full_design_space_copy, full_design_space)
    assert np.array_equal(training_inds_copy, training_inds)
    assert np.array_equal(training_fitness_copy, training_fitness)
    assert np.array_equal(training_embeddings_copy, training_embeddings)
    
    # Now run again, this time constructing the embedding indices
    constructed_training_results = run_mlde(default_mods, training_df,
                                            full_design_space, combo_to_ind,
                                            train_test_inds = train_test_inds,
                                            _reshape_x = True, _debug = True)
    
    # Make sure nothing has changed
    assert np.array_equal(full_design_space_copy, full_design_space)
    assert np.array_equal(training_inds_copy, training_inds)
    assert np.array_equal(training_fitness_copy, training_fitness)
    assert np.array_equal(training_embeddings_copy, training_embeddings)
    
    # Make sure the default model parameter values have not changed
    assert orig_default_mod_params == spd.DEFAULT_MODEL_PARAMS
    assert orig_default_train_params == spd.DEFAULT_TRAINING_PARAMS
    
    # Loop over the three different outputs
    result_names = ("training_inds", "embedded_combos", "training_fitness", 
                    "model_names", "results", "processed_results", "model_args")
    all_mlde_results = (no_mulitprocess_results, constructed_training_results) 
    for i, result_name in enumerate(result_names):
        for run_mlde_result in all_mlde_results:
                       
            # If this is "training_inds", make sure they match the input
            if result_name == "training_inds":
                assert np.array_equal(run_mlde_result[i], training_inds)
                
            # If this is "embedded_combos", make sure the outputs match the inputs
            elif result_name == "embedded_combos":
                assert np.array_equal(run_mlde_result[i], training_embeddings)
                
            # If this is training_fitness, make sure the outputs match the inputs
            elif result_name == "training_fitness":
                assert np.array_equal(run_mlde_result[i], training_fitness)
                
            # If this is model_names, make sure we match the expectation
            elif result_name == "model_names":
                assert len(expected_model_names) == len(run_mlde_result[i])
                assert all(expected_model_name == run_result for 
                           expected_model_name, run_result in 
                           zip(expected_model_names, run_mlde_result[i]))
                
            # If this is results (unprocessed, make sure the output is what
            # we expect it to be)
            elif result_name == "results":
                
                # This should be a list as long as the number of models
                assert isinstance(run_mlde_result[i], list)
                assert len(run_mlde_result[i]) == len(expected_model_names)
                
                # Loop over the individual results in the list
                for list_result in run_mlde_result[i]:
                
                    # The full result should be tuple with 4 results
                    assert isinstance(list_result, tuple)
                    assert len(list_result) == 4
                    
                    # Training loss and testing loss should be a float
                    assert isinstance(list_result[0], float)
                    assert isinstance(list_result[1], float)
                    
                    # Predictions should match the length of the full design space
                    assert len(list_result[2]) == 10000
                    assert len(list_result[3]) == 10000
                    assert len(list_result[2].shape) == 1
                    assert len(list_result[3].shape) == 1
                
            # If this is processed results, make sure the output is what we
            # expect it to be
            elif result_name == "processed_results":
                
                # The results should be a tuple of length 5
                assert isinstance(run_mlde_result[i], tuple)
                assert len(run_mlde_result[i]) == 5
                
                # The first element should be a pandas dataframe with length
                # equal to the design space
                assert isinstance(run_mlde_result[i][0], pd.DataFrame)
                assert len(run_mlde_result[i][0]) == 10000
                
                # The second element should be a pandas dataframe with length
                # equal to the number of models
                assert isinstance(run_mlde_result[i][1], pd.DataFrame)
                assert len(run_mlde_result[i][1]) == len(expected_model_names)
                
                # The third element should be a numpy array with dimensionality
                # (n_models x design_space)
                assert run_mlde_result[i][2].shape == (len(expected_model_names),
                                                       10000)
                
                # The fourth element should be a numpy array with the same
                # dimensionality as the third. Though these two arays should not
                # be equal
                assert run_mlde_result[i][3].shape == run_mlde_result[i][2].shape
                assert not np.array_equal(run_mlde_result[i][3], run_mlde_result[i][2])
                
                # The fifth element must pass the same test as the previous
                assert run_mlde_result[i][4].shape == run_mlde_result[i][3].shape
                assert not np.array_equal(run_mlde_result[i][4], run_mlde_result[i][3])
            
            # If this is gpu_args, our results are different based on whether we
            # ran multiprocessing or not
            elif result_name == "model_args":
                assert len(expected_model_names) == len(run_mlde_result[i])
                for k, (model_obj, cv_test) in enumerate(run_mlde_result[i]):
                    
                    # Make sure it is the correct model
                    assert expected_model_names[k] == f"{model_obj.major_model}-{model_obj.specific_model}"
                    
                    # Make sure all cross-validation indices are what we expect
                    assert cv_test == train_test_inds
            
            else:
                assert False
    
def test_run_hyperopt_mlde():
    
    # Build inputs for hyperopt testing
    test_training_inds = np.random.choice(np.arange(1000), 384, replace = False)
    full_space = np.random.rand(1000, 3, 20)
    training_x = full_space[test_training_inds]
    training_y = np.random.rand(384)
    model_data_bad = [[major_model, specific_model, 0] for 
                      major_model, specific_model in chain(CPU_MODELS, GPU_MODELS)]
    model_data_good = [[major_model, specific_model, 2] for 
                      major_model, specific_model in chain(CPU_MODELS, GPU_MODELS)]
    
    # Make train test inds
    splitter = KFold(5)
    test_train_test = list(splitter.split(training_x))
    
    # Make copies of input data
    full_space_copy = full_space.copy()
    training_x_copy = training_x.copy()
    training_y_copy = training_y.copy()
    
    # If we pass in the bad data, we should get a dataframe and tuple of duds
    expected_cols = ["MajorModel", "SpecificModel", "HyperRound", "RunTime",
                     "TrainErr", "TestErr", "Hyper", "HyperVal"]
    expected_bad_tuple = (np.inf, np.inf, np.zeros(1000), np.zeros(1000))
    
    # Run hyperopt with bad data and make sure results are as expected
    bad_data_test = run_hyperopt_mlde(model_data_bad, training_x, training_y,
                                      full_space, test_train_test, progress_pos = 0)
    for i, (test_df, test_tuple) in enumerate(bad_data_test):
        
        # Get the expected major and specific models
        expected_major = model_data_bad[i][0]
        expected_specific = model_data_bad[i][1]
        
        # Build the expected dataframe and tuple
        expected_bad_df = pd.DataFrame([[expected_major, expected_specific,
                                         0, 0, np.inf, np.inf, np.nan,
                                         "NoHyperoptPerformed"]],
                                       columns = expected_cols)
        
        # Make sure the dataframe output is as expected
        assert expected_bad_df.equals(test_df)
        
        # Make sure the elements of the tuple output are as expected
        assert test_tuple[0] == expected_bad_tuple[0]
        assert test_tuple[1] == expected_bad_tuple[1]
        assert np.array_equal(test_tuple[2], expected_bad_tuple[2])
        assert np.array_equal(test_tuple[3], expected_bad_tuple[3])
        
    # Run hyperopt with good data
    good_data_test = run_hyperopt_mlde(model_data_good, training_x, training_y,
                                       full_space, test_train_test, progress_pos = 0)
    
    # Make sure the outputs are the correct types and the names of models come
    # out in the input order
    for i, (test_df, test_tuple) in enumerate(good_data_test):
        
        # Get the expected majora and minor models
        expected_major = model_data_good[i][0]
        expected_specific = model_data_good[i][1]
        
        # Make sure the outputs are the appropriate type
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(test_tuple, tuple)
        
        # Make sure the major and specific model names are as expected.
        assert expected_major == test_df.MajorModel.values[0]
        assert expected_specific == test_df.SpecificModel.values[0]
        
        # Make sure the hyperparameters tested are what we expect
        expected_hypers = set(SPACE_BY_MODEL[expected_major][expected_specific])
        actual_hypers = set(test_df.Hyper.values)
        assert expected_hypers == actual_hypers
        
    # Make sure no input data changed
    assert np.array_equal(full_space_copy, full_space)
    assert np.array_equal(training_x_copy, training_x)
    assert np.array_equal(training_y_copy, training_y)
    
def test_prep_input_data():
    """
    This function confirms:
    - The correct model type is built (checking by name, model params, and 
    training params)
    - The correct model parameters are constructed for all Keras models (the 
    float parameters are correctly converted to integer parameters and the 
    correct input shape is appended
    - The correct model parameters are maintained for all other models
    - The correct information is pulled from the parameters_df    
    """
    # Load the parameter dataframe
    parameter_df = pd.read_csv("./code/validation/basic_test_data/TestMldeParams.csv")
    
    # Create a series of input shapes
    test_input_shapes = ((10000, 2, 123),
                         (1321, 5, 500),
                         (12412, 3, 421),
                         (1241, 4, 301))
    
    # What are the expected parameter values for the Keras models based on these
    # input shapes   
    expectations = (
        {
            "size1": 62,
            "size2": 16,
            "filter_choice": 1,
            "n_filters1": 8,
            "filter_arch": (1, 1),
            "n_filters2": 1 
             },
        {
            "size1": 625,
            "size2": 157,
            "filter_choice": 3,
            "n_filters1": 32,
            "filter_arch": (3, 3),
            "n_filters2": 4 
             },
        {
            "size1": 316,
            "size2": 79,
            "filter_choice": 2,
            "n_filters1": 27,
            "filter_arch": (2, 2),
            "n_filters2": 4,
             },
        {
            "size1": 301,
            "size2": 76,
            "filter_choice": 2,
            "n_filters1": 19,
            "filter_arch": (2, 2),
            "n_filters2": 3
             }
    )
    # Params to check
    params_to_check = {"size1", "size2", "filter_choice",
                       "n_filters1", "filter_arch", "n_filters2"}
    
    # Make sure that the function fails the input shape is not 3d
    with pytest.raises(AssertionError, match = "Input shape should be 3D"):
        prep_input_data(parameter_df, (3, 4))
        
    # Get the expected model names
    major_model_names = parameter_df.ModelClass.values
    specific_model_names = parameter_df.SpecificModel.values
    
    # Loop over all input shapes
    for i, shape in enumerate(test_input_shapes): 
        
        # Run prep_input_data
        model_instances, hyperopt_args = prep_input_data(parameter_df, shape)
        
        # Make sure the model instances have the correct parameters
        for j, model_instance in enumerate(model_instances):
            
            # If this is a Keras model, special treatment
            if model_instance.major_model == "Keras":
                
                # Loop over model parameters
                for key, val in model_instance.model_params.items():
                    
                    # If this model parameter is one to check, do that now
                    if key in params_to_check:
                        assert expectations[i][key] == val
                
                    # Make sure that the input shape is correct
                    elif key == "input_shape":
                        
                        # If this is not a convolutional, shape should be the
                        # product of the last dimensions. Otherwise, it should
                        # just be the last dimensions
                        if model_instance.specific_model in {"OneConv", "TwoConv"}:
                            assert val == shape[1:]
                        else:
                            assert val == (shape[1] * shape[2],)
                        
                    # Otherwise, make sure it matches what's in the default
                    # parameters
                    else:
                        assert val == DEFAULT_MODEL_PARAMS["Keras"][model_instance.specific_model][key]
            
            # If this is not a keras model, make sure that the returned parameters
            # match what's in the default parameters
            else:
                assert (model_instance.model_params == 
                        DEFAULT_MODEL_PARAMS[model_instance.major_model][model_instance.specific_model])
            
            # Make sure the model name matches the expected
            assert major_model_names[j] == model_instance.major_model
            assert specific_model_names[j] == model_instance.specific_model
            
            # Check the hyperopt arguments
            assert major_model_names[j] == hyperopt_args[j][0]
            assert specific_model_names[j] == hyperopt_args[j][1]
            assert 2 == hyperopt_args[j][2]

def test_run_mlde_cl():
    """
    Things to test....
    1. Does making "shuffle" == False actually stop shuffling?
    2. Does including too few models in the inclusion frame throw a warning?
    
    Everything else is a subfunction which is tested independently. We know that
    this function gives reasonable outputs, so the subfunctions are presumably
    playing well together. We just need to test this higher level information 
    then.
    """
    
    # Load the test dataframe
    test_df = pd.read_csv("./code/validation/basic_test_data/InputValidationData.csv")
    design_space = np.load("./code/validation/basic_test_data/GB1_T2Q_georgiev_Normalized.npy")
    with open("./code/validation/basic_test_data/GB1_T2Q_ComboToIndex.pkl", "rb") as f:
        combo_to_ind = pickle.load(f)
    test_params = pd.read_csv("./code/validation/pytest/run/mlde_parameters.csv")
    
    # Define the expected models in the correct order
    expected_model_names = np.array(["Keras-NoHidden",
                                     "Keras-TwoHidden",
                                     "Keras-OneConv",
                                     "XGB-Tree",
                                     "XGB-Linear",
                                     "XGB-Tree-Tweedie",
                                     "sklearn-regressor-Linear",
                                     "sklearn-regressor-GradientBoostingRegressor",
                                     "sklearn-regressor-RandomForestRegressor",
                                     "sklearn-regressor-BayesianRidge",
                                     "sklearn-regressor-LinearSVR",
                                     "sklearn-regressor-ARDRegression",
                                     "sklearn-regressor-KernelRidge",
                                     "sklearn-regressor-BaggingRegressor",
                                     "sklearn-regressor-DecisionTreeRegressor",
                                     "sklearn-regressor-SGDRegressor",
                                     "sklearn-regressor-KNeighborsRegressor",
                                     "sklearn-regressor-ElasticNet"])

    # Copy inputs to make sure they don't change
    design_space_copy = design_space.copy()
    combo_to_ind_copy = combo_to_ind.copy()
    test_df_copy = test_df.copy()
    test_params_copy = test_params.copy()
    
    # Run with shuffle off. Make sure we get a warning.
    with pytest.warns(UserWarning, match = "Requested averaging 100, but only 18 will be trained. Averaging all models."):
        test_model_names, train_test_inds = run_mlde_cl(test_df, design_space, combo_to_ind, test_params, "",
                                                        n_to_average = 100, shuffle = False, _debug = True)
    
    # Make sure we load the correct models
    assert np.array_equal(test_model_names, expected_model_names)
    
    # Make sure that nothing was shuffled 
    counter = 0
    assert len(train_test_inds) == 5
    for train_inds, test_inds in train_test_inds:
        for test_ind in test_inds:
            assert counter == test_ind
            counter += 1
        
    # Now run with shuffle on. Lower the n_to_average so that we do not get a
    # warning. Also change the n_cv to make sure this is changed in the output.
    test_model_names2, train_test_inds2 = run_mlde_cl(test_df, design_space, combo_to_ind, test_params, "",
                                                        n_to_average = 5, n_cv = 3, shuffle = True, _debug = True)
    
    # Make sure we load the correct models
    assert np.array_equal(test_model_names2, expected_model_names)
    
    # Make sure that everything was shuffled 
    counter = 0
    in_order = True
    assert len(train_test_inds2) == 3
    for train_inds, test_inds in train_test_inds2:
        for test_ind in test_inds:
            if counter != test_ind:
                in_order = False
            counter += 1
    assert not in_order
    
    # Make sure no inputs changed
    assert np.array_equal(design_space_copy, design_space)
    assert combo_to_ind_copy == combo_to_ind
    assert test_df_copy.equals(test_df)
    assert test_params_copy.equals(test_params)
        
def test_combine_results():
    """
    Test to be sure that we are successfully combining results from default
    MLDE and hyperopt MLDE. This function should combine results from MLDE
    """
    
    # Make fake hyperopt results. This is a list of trial dfs and train-pred
    # outputs
    hyperopt_cols = ["MajorModel", "SpecificModel", "HyperRound", "RunTime",
                     "TrainErr", "TestErr", "Hyper", "HyperVal"]
    fake_hyperopt = (
        (pd.DataFrame((("Test1", "Test2", 0, 231.1, 0.01, 0.05, "TestHyp1", True),
                      ("Test1", "Test2", 1, 31.1, 0.09, 0.8, "TestHyp1", False)),
                      columns = hyperopt_cols), 
         (0.02, 0.07, np.array([1, 3, 4]), np.array([4, 9, 102.1]))),
        (pd.DataFrame((("Test3", "Test4", 0, 231.9, 1, 0.98, "TestHyp2", 1),
                      ("Test3", "Test4", 10, 3121.1, 0.079, 0.809, "TestHyp2", 0.9)),
                      columns = hyperopt_cols), 
         (0.02, 0.09, np.array([98.1, 13, 12.3]), np.array([2.1, -0.9, -15])))
        )
    
    # Make fake run_mlde results. This should just be a stack of results from train
    # and predict
    fake_default = ((0.01, 0.09, np.array([5, 8, 9]), np.array([10, 11, 15])),
                    (0.06, 0.06, np.array([7, 8, 10]), np.array([14, 17, -1.2])))
    
    # Build the expected output
    expected_df = pd.DataFrame((("Test1", "Test2", 0, 231.1, 0.01, 0.05, "TestHyp1", True),
                                ("Test1", "Test2", 1, 31.1, 0.09, 0.8, "TestHyp1", False),
                                ("Test3", "Test4", 0, 231.9, 1, 0.98, "TestHyp2", 1),
                                ("Test3", "Test4", 10, 3121.1, 0.079, 0.809, "TestHyp2", 0.9)),
                               columns = hyperopt_cols)
    expected_output = ((0.02, 0.07, np.array([1, 3, 4]), np.array([4, 9, 102.1])),
                       (0.06, 0.06, np.array([7, 8, 10]), np.array([14, 17, -1.2])))
    
    # Run combine results
    true_output, true_output_df = combine_results(fake_default, fake_hyperopt, 
                                                  "output", _debug = True)
    
    # Make sure the concatenated dataframe is as expected
    assert np.array_equal(expected_df.values, true_output_df.values)
    
    # Make the combined results are as we expect
    for expected_row, true_row in zip(expected_output, true_output):
        assert expected_row[0] == true_row[0]
        assert expected_row[1] == true_row[1]
        assert np.array_equal(expected_row[2], true_row[2])
        assert np.array_equal(expected_row[3], true_row[3])