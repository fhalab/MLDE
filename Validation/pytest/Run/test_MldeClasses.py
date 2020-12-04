"""
This script handles testing the MldeClasses objects
"""
# Load relevant information
from Support.RunMlde.MldeClasses import (MldeModel, KerasModel, 
                                         XgbModel, SklearnRegressor)
from Support.Params.Defaults import (cpu_models, gpu_models,
                                     default_model_params, 
                                     default_training_params)
from Support.RunMlde.CompleteKerasParams import process_val
from Support.RunMlde.LossFuncs import mse
import pytest
import numpy as np
import xgboost as xgb
from itertools import chain
from sklearn.model_selection import KFold

# Filter convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Load the most informative samples
most_inform_inds = np.load("/home/brucejwittmann/GitRepos/MLDE/Simulate/InputData/CadexIndices/Transformer.npy")
all_x = np.load("/home/brucejwittmann/GitRepos/MLDE/Simulate/InputData/Encodings/Transformer.npy")
all_y = np.load("/home/brucejwittmann/GitRepos/MLDE/Simulate/InputData/Fitness.npy")

# Take the first 384 indices of the most informative
train_inds = most_inform_inds[:300]
test_inds = most_inform_inds[300:384]

# Make a flat all_x
flat_all_x = np.reshape(all_x, [len(all_x), 4*512])

# Make fake data to pass in. We will need both 2D and 3D data.
x_train2d = flat_all_x[train_inds]
x_test2d = flat_all_x[test_inds]
x_train3d = all_x[train_inds]
x_test3d = all_x[test_inds]
y_train = all_y[train_inds]
y_test = all_y[test_inds]

# Make copies of all of the fake data
most_inform_inds_copy = most_inform_inds.copy()
all_x_copy = all_x.copy()
all_y_copy = all_y.copy()

# Take the first 384 indices of the most informative
train_inds_copy = train_inds.copy()
test_inds_copy = test_inds.copy()

# Make a flat all_x
flat_all_x_copy = flat_all_x.copy()

# Make fake data to pass in. We will need both 2D and 3D data.
x_train2d_copy = x_train2d.copy()
x_test2d_copy = x_test2d.copy()
x_train3d_copy = x_train3d.copy()
x_test3d_copy = x_test3d.copy()
y_train_copy = y_train.copy()
y_test_copy = y_test.copy()

# Define the expected parameters for the submodels
# Write in the expected model params for each model
expected_model_params_xgb = {"Linear": {"booster": "gblinear",
                                    "tree_method": "exact",
                                    "nthread": 1,
                                    "verbosity": 0,
                                    "objective": "reg:squarederror",
                                    "eval_metric": "rmse",
                                    "lambda": 1,
                                        "alpha": 0},
                            "Tree": {"booster": "gbtree",
                                    "tree_method": "exact",
                                    "nthread": 1,
                                    "verbosity": 0,
                                    "objective": "reg:squarederror",
                                    "eval_metric": "rmse",
                                    "eta": 0.3,
                                    "max_depth": 6,
                                    "lambda": 1,
                                    "alpha": 0},
                            "Linear-Tweedie": {"booster": "gblinear",
                                            "tree_method": "exact",
                                            "nthread": 1,
                                            "verbosity": 0,
                                            "objective": "reg:tweedie",
                                            "tweedie_variance_power": 1.5,
                                            "eval_metric": "tweedie-nloglik@1.5",
                                            "lambda": 1,
                                            "alpha": 0},
                            "Tree-Tweedie": {"booster": "gbtree",
                                            "tree_method": "exact",
                                            "nthread": 1,
                                            "verbosity": 0,
                                            "objective": "reg:tweedie",
                                            "tweedie_variance_power": 1.5,
                                            "eval_metric": "tweedie-nloglik@1.5",
                                            "eta": 0.3,
                                            "max_depth": 6,
                                            "lambda": 1,
                                            "alpha": 0}}
expected_training_params_xgb = {"early_stopping_rounds": 10,
                                "num_boost_round": 1000,
                                "verbose_eval": False}

# Get the expected training parameters
expected_training_params_keras = {"patience": 10, 
                            "batch_size": 32,
                            "epochs": 1000}

# Define the expected names of the sklearn models
expected_sklearnnames = {"Linear": "LinearRegression",
                        "GradientBoostingRegressor": "GradientBoostingRegressor",
                        "RandomForestRegressor": "RandomForestRegressor",
                        "LinearSVR": "LinearSVR",
                        "ARDRegression": "ARDRegression",
                        "KernelRidge": "KernelRidge",
                        "BayesianRidge": "BayesianRidge",
                        "BaggingRegressor": "BaggingRegressor",
                        "LassoLarsCV": "LassoLarsCV",
                        "DecisionTreeRegressor": "DecisionTreeRegressor",
                        "SGDRegressor": "SGDRegressor",
                        "KNeighborsRegressor": "KNeighborsRegressor",
                        "ElasticNet": "ElasticNet",
                        "AdaBoostRegressor": "AdaBoostRegressor"}

# Write a function that tests predictions of subclasses
def subclass_pred_checker(test_model, x_train, y_train, x_test, 
                          y_test, xgboost = False, tree = False):
    """
    What this function checks:
    1) Are predictions made by a submodel (From MldeModel) the correct shape? Do
    they have the same length as the input x? Are they all 1d?
    2) Are predictions made by a submodel consistent? Does calling predict multiple
    times give the same answer?
    3) Do the extracted models (by calling an attribute) give the same predictions
    as the Mlde submodels themselves
    """
    # Copy training data
    local_x_train_copy = x_train.copy()
    local_x_test_copy = x_test.copy()
    local_y_train_copy = y_train.copy()
    local_y_test_copy = y_test.copy()
    
    # Train the model for real, then confirm that prediction returns what we
    # would expect (in terms of shape)
    test_model.train(x_train, y_train, x_test, y_test)
    test_preds1 = test_model.predict(x_test)
    test_preds2 = test_model.predict(x_train)
    assert len(test_preds1) == len(x_test)
    assert len(test_preds2) == len(x_train)
    assert len(test_preds1.shape) == 1
    assert len(test_preds2.shape) == 1
    
    # Make sure nothing changed
    assert np.array_equal(local_x_train_copy, x_train)
    assert np.array_equal(local_x_test_copy, x_test)
    assert np.array_equal(local_y_train_copy, y_train)
    assert np.array_equal(local_y_test_copy, y_test)
    assert np.array_equal(most_inform_inds_copy, most_inform_inds)
    assert np.array_equal(all_x_copy, all_x)
    assert np.array_equal(all_y_copy, all_y)
    assert np.array_equal(train_inds_copy, train_inds)
    assert np.array_equal(test_inds_copy, test_inds)
    assert np.array_equal(flat_all_x_copy, flat_all_x)
                
    # Confirm that predictions are consistent
    repeat_test_preds1 = test_model.predict(x_test)
    repeat_test_preds2 = test_model.predict(x_train)
    assert np.array_equal(test_preds1, repeat_test_preds1)
    assert np.array_equal(test_preds2, repeat_test_preds2)
    
    # Make sure the extracted model predicts the same as the real
    mod_property = test_model.mod
    if xgboost and tree:
        final_test_preds1 = mod_property.predict(xgb.DMatrix(x_test), 
                                                             ntree_limit = mod_property.best_ntree_limit)
        final_test_preds2 = mod_property.predict(xgb.DMatrix(x_train),
                                                             ntree_limit = mod_property.best_ntree_limit)
    elif xgboost:
        final_test_preds1 = mod_property.predict(xgb.DMatrix(x_test))
        final_test_preds2 = mod_property.predict(xgb.DMatrix(x_train))
    else:
        final_test_preds1 = mod_property.predict(x_test)
        final_test_preds2 = mod_property.predict(x_train)
    assert np.array_equal(final_test_preds1.flatten(), test_preds1)
    assert np.array_equal(final_test_preds2.flatten(), test_preds2)

# Write tests for each major class
def test_KerasModel():
    """
    What this function confirms:
    1) Passing the wrong set of parameters into a Keras model (e.g. passing in
    the parameters for OneHidden into OneConv) will result in an error
    2) Attempting to train with more or less points than labels  will result in
    an error
    3) All checks in subclass_pred_checker() are passed by KerasModel()
    4) Training parameters are appropriately handled by KerasModel
    """
    # Associate a class method with each specific model
    class_method_dict = {"NoHidden": KerasModel.NoHidden,
                         "OneHidden": KerasModel.OneHidden,
                         "TwoHidden": KerasModel.TwoHidden,
                         "OneConv": KerasModel.OneConv,
                         "TwoConv": KerasModel.TwoConv}
       
    # Define training aprams
    training_params = default_training_params["Keras"].copy()
              
    # Make sure that passing the parameters in for the wrong model throws 
    # an error
    for major_model, specific_model in gpu_models:
        
        # Pull the parameters
        model_params = default_model_params[major_model][specific_model].copy()
        
        # Add input_shape
        if specific_model in {"OneConv", "TwoConv"}:
            model_params["input_shape"] = all_x.shape[1:]
            x_train = x_train3d
            x_test = x_test3d
        else:
            model_params["input_shape"] = flat_all_x.shape[1:]
            x_train = x_train2d
            x_test = x_test2d
            
        # Copy train and test
        local_x_train_copy = x_train.copy()
        local_x_test_copy = x_test.copy()
            
        # Complete model parameters
        model_params = {key: process_val(key, val, x_train.shape)
                        for key, val in model_params.items()}
        
        # An error should be raised if we pass the wrong parameters to a model
        for _, other_specific in gpu_models:
            
            # Continue if this is the same model
            if other_specific == specific_model:
                continue
            
            # Confirm that we get an error if passing in the wrong set of parameters
            with pytest.raises(AssertionError, match = "(Some model_params missing for .+|Too many parameters passed .+)"):
                _ = class_method_dict[other_specific](model_params,
                                                     training_params)
                
        # Build a test model
        test_model = class_method_dict[specific_model](model_params, 
                                                       training_params)
        
        # If we try to train on data with mismatched sizes we should run into
        # an error
        with pytest.raises(AssertionError, match = "Mismatch in lengths of .+"):
            test_model.train(x_train, y_test, x_test, y_train)
            
        # Make sure nothing changes
        assert np.array_equal(local_x_train_copy, x_train)
        assert np.array_equal(local_x_test_copy, x_test)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
                    
        # Test predictions
        subclass_pred_checker(test_model, x_train, y_train, x_test, y_test)
        
        # Make sure we fail if we try to train multiple times
        with pytest.raises(AssertionError, match = "Successive calls to 'train' not supported"):
            test_model.train(x_train, y_train, x_test, y_test)
            
        # Make sure nothing changes
        assert np.array_equal(local_x_train_copy, x_train)
        assert np.array_equal(local_x_test_copy, x_test)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
        
        # Confirm that all properties return something and that they match
        # the model instance variable
        test_early_stop = test_model.early_stopping_epoch
        test_training_params = test_model.training_params
        assert test_early_stop == test_model._early_stopping_epoch
        assert test_training_params == test_model._training_params
        
        # Make sure we have all expected training params
        assert test_training_params == expected_training_params_keras
                
        # Make sure that the early stopping epoch is an integer
        assert isinstance(test_early_stop, int)
        
# Write a test for XgbModel
def test_XgbModel():
    """
    What this function confirms:
    1) Model and training parameters are appropriately handled by XgbModel()
    2) Attempting to train with more or less points than labels  will result in
    an error
    3) All checks in subclass_pred_checker() are passed by XgbModel()
    4) Training parameters are appropriately handled by XgbModel()
    5) Attempting to train or predict using a 3D matrix will result in an error
    """
    # Associate a class method with each specific model
    class_method_dict = {"Linear": XgbModel.Linear,
                         "Tree": XgbModel.Tree,
                         "Linear-Tweedie": XgbModel.LinearTweedie,
                         "Tree-Tweedie": XgbModel.TreeTweedie}                        
    
    # Get the training params
    training_params = default_training_params["XGB"].copy()
    
    # Loop over all model types
    for model_type in class_method_dict.keys():
               
        # Identify the default model params
        mod_params = default_model_params["XGB"][model_type]
                
        # Build the model
        constructed_mod = class_method_dict[model_type](mod_params, 
                                                        training_params)
        
        # Extract model and training params
        extracted_training_params = constructed_mod.training_params
        extracted_model_params = constructed_mod.model_params
        
        # Make sure the property is equal to the instance variable
        assert extracted_training_params == constructed_mod._training_params
        assert extracted_model_params == constructed_mod._model_params
        
        # Make sure the training params and model params match the expected
        assert extracted_training_params == expected_training_params_xgb
        assert extracted_model_params == expected_model_params_xgb[model_type]
        
        # Determine if this is a tree model
        tree = True if extracted_model_params["booster"] == "gbtree" else False
        
        # If we try to train on data with mismatched sizes we should run into
        # an error
        with pytest.raises(AssertionError, match = "Mismatch in lengths of .+"):
            constructed_mod.train(x_train2d, y_test, x_test2d, y_train)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
            
        # If we try to train with a 3D matrix we should get an error
        with pytest.raises(AssertionError, match = "x values must be a 2D matrix"):
            constructed_mod.train(x_train3d, y_train, x_test3d, y_test)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
            
        # Test predictions
        subclass_pred_checker(constructed_mod, x_train2d, y_train, 
                              x_test2d, y_test, xgboost = True, tree = tree)
        
        # Make sure the early stopping epoch is an intenger and that the 
        # extracted matches the actual
        assert isinstance(constructed_mod.early_stopping_epoch, int)
        assert constructed_mod.early_stopping_epoch == constructed_mod._early_stopping_epoch
        
        # Throw an error if you try to predict with a 3D x
        with pytest.raises(AssertionError, match = "Expected a 2D input for x"):
            constructed_mod.predict(x_test3d)
            
        # Make sure nothing changes
        assert np.array_equal(x_test3d, x_test3d)

# Write a function that tests the SklearnRegressor submodel
def test_SklearnRegressor():
    """
    What this function confirms:
    - Attempting to train with more or less points than labels  will result in
    an error
    - All checks in subclass_pred_checker() are passed by XgbModel()
    - Attempting to train or predict using a 3D matrix will result in an error
    """
    # Package the class methods
    all_class_methods = {"Linear": SklearnRegressor.Linear,
                         "GradientBoostingRegressor": SklearnRegressor.GradientBoostingRegressor,
                         "RandomForestRegressor": SklearnRegressor.RandomForestRegressor,
                         "LinearSVR": SklearnRegressor.LinearSVR,
                         "ARDRegression": SklearnRegressor.ARDRegression,
                         "KernelRidge": SklearnRegressor.KernelRidge,
                         "BayesianRidge": SklearnRegressor.BayesianRidge,
                         "BaggingRegressor": SklearnRegressor.BaggingRegressor,
                         "LassoLarsCV": SklearnRegressor.LassoLarsCV,
                         "DecisionTreeRegressor": SklearnRegressor.DecisionTreeRegressor,
                         "SGDRegressor": SklearnRegressor.SGDRegressor,
                         "KNeighborsRegressor": SklearnRegressor.KNeighborsRegressor,
                         "ElasticNet": SklearnRegressor.ElasticNet}
       
    # Loop over all model types
    for model_name in all_class_methods.keys():
        
        # Build the model
        model_params = default_model_params["sklearn-regressor"][model_name].copy()
        test_model = all_class_methods[model_name](model_params)

        # If we try to train on data with mismatched sizes we should run into
        # an error
        with pytest.raises(AssertionError, match = "Mismatch in lengths of .+"):
            test_model.train(x_train2d, y_test, x_test2d, y_train)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
                    
        # Test predictions
        subclass_pred_checker(test_model, x_train2d, y_train, x_test2d, y_test)
        
        # If we try and pass in a 3D x to train we should fail
        with pytest.raises(AssertionError, match = "x values must be a 2D matrix"):
            test_model.train(x_test3d, y_test, x_train3d, y_train)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
        
        # If we try and pass in a 3D x to predict we should fail
        with pytest.raises(AssertionError, match = "x must be 2D"):
            test_model.predict(x_test3d)
            
        # Make sure nothing changes
        assert np.array_equal(x_test3d, x_test3d)

# Write a function that confirms an MldeModel instance has constructed the 
# correct submodel
def confirm_correct_built_params(test_mlde_model, test_submodel,
                                 major_model, specific_model):
    """
    What this function checks:
    1) If the major model is an sklearn-regressor, the expected sklearn model
    is loaded as the submodel
    2) If the model is an Xgboost model, the expected model and training 
    parameters are used
    3) If this is a Keras model, the expected training parameters are used (
    the checks here are less stringent as we have inbuilt parameter checking 
    into the Keras submodels already)
    """
    # If sklearn, confirm that the model name is what we expect
    if major_model == "sklearn-regressor":
        
        # Get the name of instantiated sklearn model
        sklearn_model_name = type(test_submodel.mod).__name__
        
        # Make sure that the name matches the expected
        assert sklearn_model_name == expected_sklearnnames[specific_model]
        
    # If xgboost, confirm that the model parameters and training parameters
    # are what we expect
    elif major_model == "XGB":
    
        # The extracted model should match the params
        assert expected_model_params_xgb[specific_model] == test_submodel.model_params
        assert expected_training_params_xgb == test_submodel.training_params
        
        # Parent model should also have matching training params (model params
        # will differ depending on the submodel called)
        assert expected_training_params_xgb == test_mlde_model.training_params
        
    # If Keras, confirm that the training parameters are what we expect
    elif major_model == "Keras":
        assert expected_training_params_keras == test_submodel.training_params
        assert expected_training_params_keras == test_mlde_model.training_params

# Write a function to complete keras params
def temporary_complete_keras(major_model, specific_model, model_params):
    """
    This function is designed only to complete Keras parameters. It wraps 
    process_val() to make sure that the non-integer parameters are correctly
    translated prior to being used to build keras models
    """
    # If a keras model, add to model params
    if major_model == "Keras":
        if specific_model in {"OneConv", "TwoConv"}:
            model_params["input_shape"] = x_test3d.shape[1:]
            model_params = {key: process_val(key, val, x_test3d.shape)
                            for key, val in model_params.items()}
        else:
            model_params["input_shape"] = x_test2d.shape[1:]
            model_params = {key: process_val(key, val, x_test2d.shape)
                            for key, val in model_params.items()}
            
    return model_params

# Write a function that tests MldeModel._build_model
def test_MldeModel_build_model():
    """
    What this function confirms:
    1) MldeModel._build_model() builds the correct model based on the input
    major and specific models. This is confirmed by running the function 
    "confirm_correct_built_params" for each inbuilt MldeModel
    """
    # Loop over all model types
    for major_model, specific_model in chain(gpu_models, cpu_models):
        
        # Pull the training and model parameters
        training_params = default_training_params[major_model].copy()
        model_params = default_model_params[major_model][specific_model].copy()

        # Complete the keras parameters
        model_params = temporary_complete_keras(major_model, specific_model, 
                                                model_params)

        # Instantiate an MldeModel
        test_mlde_model = MldeModel(major_model, specific_model, 
                               model_params = model_params,
                               training_params = training_params)
        
        # Build a model and return it
        test_submodel = test_mlde_model._build_model()
        
        # Confirm that we have all of the expected parameters
        confirm_correct_built_params(test_mlde_model, test_submodel,
                                     major_model, specific_model)
            
# Write a function that tests all other functions associated with MldeModel
def test_MldeModel():
    """
    What this function confirms:
    - Mean and standard deviation of the cross-validation predictions are 
    calculated over the correct dimensions
    - Different submodels in an MldeModel give different predictions (as evident
    by a non-zero standard deviation)
    - The passed in eval metric is correctly used by MldeModel
    - If a list is passed in as n_cv rather than an int, then it is what is used
    for the cross-validation inds
    - If we pass in train-test inds that either do not cover all of x_train or 
    have more indices than are in x_train, then we throw an error 
    - If there are duplicate indices in train-test inds, then we throw an error
    - MldeModel.clear_submodels() deletes all models
    - If we don't pass in a list of train-test inds, then we build one based on
    n_cv
    """
    # Loop over all model types
    for major_model, specific_model in chain(gpu_models, cpu_models):
        
        # Get training and testing parameters
        training_params = default_training_params[major_model].copy()
        model_params = default_model_params[major_model][specific_model].copy()
        
        # Complete model params
        model_params = temporary_complete_keras(major_model, specific_model,
                                                model_params)
        
        # Instantiate a test model
        test_mlde_model = MldeModel(major_model, specific_model,
                                    model_params = model_params,
                                    training_params = training_params)
        
        # Make some train-test inds
        splitter = KFold(n_splits = 5, shuffle = True, random_state=2)
        split_inds = list(splitter.split(x_train2d))
        
        # Train the model
        if major_model == "Keras" and specific_model in {"OneConv", "TwoConv"}:        
            tt_inds, train_loss, test_loss = test_mlde_model.train_cv(x_train3d, 
                                                                      y_train,
                                                                      split_inds,
                                                                      _debug = True)
            x_test = x_test3d
        else:
            tt_inds, train_loss, test_loss = test_mlde_model.train_cv(x_train2d, 
                                                                      y_train,
                                                                      split_inds,
                                                                      _debug = True)
            x_test = x_test2d
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
            
        # Make sure predictions are coming back the appropriate shape 
        mean_preds, stdev_preds = test_mlde_model.predict(x_test)
        assert len(mean_preds) == len(x_test)
        assert len(stdev_preds) == len(x_test)
        assert len(mean_preds.shape) == 1
        assert len(stdev_preds.shape) == 1
        
        # Make sure the predictions are different (this signifies that we are
        # using a different model for each round of training)
        assert not np.allclose(stdev_preds, np.zeros(len(stdev_preds)))
        
        # Loop over all models built during training
        for i, trained_mod in enumerate(test_mlde_model._models):
            
            # Make sure the correct models are being called during training
            confirm_correct_built_params(test_mlde_model, trained_mod,
                                         major_model, specific_model)
            
            # Get predictions for the current model
            current_preds = trained_mod.predict(x_test)
            
            # Make sure nothing changes
            assert np.array_equal(x_train3d_copy, x_train3d)
            assert np.array_equal(x_test3d_copy, x_test3d)
            assert np.array_equal(x_train2d_copy, x_train2d)
            assert np.array_equal(x_test2d_copy, x_test2d)
            assert np.array_equal(y_train_copy, y_train)
            assert np.array_equal(y_test_copy, y_test)
            assert np.array_equal(all_x_copy, all_x)
            assert np.array_equal(all_y_copy, all_y)
            assert np.array_equal(train_inds_copy, train_inds)
            assert np.array_equal(test_inds_copy, test_inds)
            assert np.array_equal(flat_all_x_copy, flat_all_x)
            
            # Make sure all models give different predictions
            for j, other_mod in enumerate(test_mlde_model._models):
                
                # Continue if i == j
                if i == j:
                    continue
                assert not np.array_equal(current_preds, other_mod.predict(x_test))
                
            # Confirm that we are correctly calculating loss from the eval metric
            assert (mse(y_test, current_preds) == 
                    test_mlde_model._eval_metric(y_test, current_preds))
                    
        # Confirm that we use the train_test_inds given in a list if that is what's
        # passed in to train_cv
        assert isinstance(tt_inds, list)
        for i, (test_train_inds, test_test_inds) in enumerate(split_inds):
            
            # Make sure nothing changed with the input cross-val inds
            assert np.array_equal(test_train_inds, split_inds[i][0])
            assert np.array_equal(test_test_inds, split_inds[i][1])
        
        # Make sure we fail if the train-test-inds passed in do not match
        # the length of x
        with pytest.raises(AssertionError, match = "Cross val ind error"):
            splitter = KFold(n_splits = 5, shuffle = True)
            split_inds = list(splitter.split(x_train2d))
            _ = test_mlde_model.train_cv(x_test, y_test, split_inds)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
            
        # Make sure we fail if there are duplicate indices in the train-test
        # split
        with pytest.raises(AssertionError, match = "Duplicate cross val inds identified"):
            splitter = KFold(n_splits = 5, shuffle = True)
            split_inds = list(splitter.split(x_test))
            split_inds[0][0][1] = 1
            split_inds[0][0][2] = 1
            _ = test_mlde_model.train_cv(x_test, y_test, split_inds)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
            
        # Make sure clear_submodels is clearing appropriately
        assert hasattr(test_mlde_model, "_models")
        test_mlde_model.clear_submodels()
        assert not hasattr(test_mlde_model, "_models")
        
        # If we don't pass a list to n_cv, then we should not have an error
        # a list come back
        with pytest.raises(AssertionError, match = "Expected cross-validation indices to be a list"):
            tt_inds, train_loss, test_loss = test_mlde_model.train_cv(x_test, 
                                                                    y_test,
                                                                    np.array([]),
                                                                    _debug = True)
            
        # Make sure nothing changes
        assert np.array_equal(x_train3d_copy, x_train3d)
        assert np.array_equal(x_test3d_copy, x_test3d)
        assert np.array_equal(x_train2d_copy, x_train2d)
        assert np.array_equal(x_test2d_copy, x_test2d)
        assert np.array_equal(y_train_copy, y_train)
        assert np.array_equal(y_test_copy, y_test)
        assert np.array_equal(all_x_copy, all_x)
        assert np.array_equal(all_y_copy, all_y)
        assert np.array_equal(train_inds_copy, train_inds)
        assert np.array_equal(test_inds_copy, test_inds)
        assert np.array_equal(flat_all_x_copy, flat_all_x)
    