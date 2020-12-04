"""
This file contains a single function for training and predicting from a single
instantiated MldeModel
"""
# Import necessary modules
from Support.RunMlde.FinalizeX import finalize_x
import numpy as np
import warnings

# Write a function that trains and predicts using an MLDE model
def train_and_predict(model, sampled_x = None, sampled_y = None, 
                      x_to_predict = None, train_test_inds = None,
                      _reshape_x = False, _debug = False):
    """
    Function for training and then predicting from an MldeModel.
    
    Parameters
    ----------
    model: MLDE.Support.RunMlde.MldeClasses.MldeModel instance
        Instantiated MldeModel instance
    sampled_x: numpy array: default = None
        Training features
    sampled_y: numpy array, 1D: default = None
        Training labels
    x_to_predict: numpy array: default = None
        Features for which we want to predict labels
    train_test_inds: list of lists: default = None
        Cross validation indices to use in training.
        
    Returns
    -------
    training_loss: float
        Cross-validation training loss
    testing_loss: float
        Cross-validation testing loss
    mean_preds: 1D numpy array
        Mean predictions by the n_cv models generated in training
    stdev_preds: 1D numpy array
        Standard deviation of predictions made by the n_cv models generated 
        in training    
    """
    # Make sure that input x and y values are appropriate
    assert sampled_x.shape[1:] == x_to_predict.shape[1:], "Mismatch in training and prediction dimensionality"
    assert len(sampled_x) == len(sampled_y), "Mismatch in number of labels and number of training points"
    assert len(sampled_y.shape) == 1, "y should be 1D"
    
    # Finalize x shape
    if _reshape_x:
        sampled_x = finalize_x(model.major_model, model.specific_model, sampled_x)
        x_to_predict = finalize_x(model.major_model, model.specific_model, x_to_predict)

    # Wrap everything in a try-except
    try:

        # Train the model and record training/testing error
        if _debug:
            (train_test_inds, training_loss,
              testing_loss) = model.train_cv(sampled_x, sampled_y, train_test_inds,
                                             _debug = True)
        else:
            training_loss, testing_loss = model.train_cv(sampled_x, sampled_y, 
                                                         train_test_inds)
        # Predict against other_x using the trained model
        mean_preds, stdev_preds = model.predict(x_to_predict)
        
        # Clear the model
        model.clear_submodels()
        
        # Return all relevant information
        if _debug and _reshape_x:
            return (training_loss, testing_loss, mean_preds, stdev_preds, 
                    sampled_x, x_to_predict, train_test_inds)
        else:
            return training_loss, testing_loss, mean_preds, stdev_preds

    except Exception as e:
        
        # Warn user
        warnings.warn(f"Error when training {model.major_model}-{model.specific_model}: {e}.")
        
        # Get the number of expected predictions
        expected_n_preds = len(x_to_predict)
        
        # Build a set of fake train_pred_results
        train_pred_results = (np.inf, np.inf,
                              np.zeros(expected_n_preds),
                              np.zeros(expected_n_preds))
        
        return train_pred_results
