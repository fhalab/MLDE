def check_training_inputs(x_train, y_train, x_test, y_test):
    
    # Assert that x_train and y_train are the same length and that x_test and
    # y_test are the same length
    assert len(x_train) == len(y_train), "Mismatch in lengths of training labels and features"
    assert len(x_test) == len(y_test), "Mismatch in lengths of test labels and features"
    
    # Make sure that xs have the same dimensionality after the first
    assert x_train.shape[1:] == x_test.shape[1:], "x_train and x_test have different shapes" 
    
    # Make sure y is always 1d
    assert len(y_train.shape) == 1, "y_train should be 1d"
    assert len(y_test.shape) == 1, "y_train should be 1d"
    
def check_keras_model_params(expected_params, model_params, model_name):
    
    # Identify the missing params
    missing_params = [param for param in expected_params if param not in model_params]
    assert len(missing_params) == 0, f"Some model_params missing for {model_name}: {missing_params}"
    
    # Identify additional params
    additional_params = [param for param in model_params if param not in expected_params]
    assert len(additional_params) == 0, f"Too many parameters passed in to {model_name}: {additional_params}"