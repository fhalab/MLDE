"""
This file contains the classes which are the center of all MLDE calculations. On 
import, this module loads:

Classes
-------
MldeModel: Highest level class for performing MLDE operations
KerasModel: Container for keras models
XgbModel: Container for XGBoost models
SklearnRegressor: Container for sklearn regressor models
"""

# Filter convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Import sklearn regression objects
from sklearn.linear_model import (ARDRegression, BayesianRidge, LassoLarsCV,
                                  SGDRegressor, ElasticNet, LinearRegression)
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              BaggingRegressor, AdaBoostRegressor)
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Import keras objects and tensorflow
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (GlobalAveragePooling1D, GlobalMaxPooling1D,
                          BatchNormalization, Dense, Activation,
                          Dropout, Flatten, Conv1D)
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session

# Import all other packages
import numpy as np
import xgboost as xgb
from copy import deepcopy
import gc
import os

# Import MLDE packages
from .loss_funcs import mse
from .input_check import check_training_inputs, check_keras_model_params

# Configure tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
_ = set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
              
# Write a parent class that can train and predict using the smaller model classes
class MldeModel():
    """
    The main class for performing MLDE operations. Handles training and prediction
    for a given model architecture.
    
    Parameters
    ---------
    major_model: str
        Choice of 'Keras', 'XGB', 'sklearn-regressor', or 'Custom'. This argument
        tells MldeModel from which package we will be pulling models. 'Custom'
        allows the user to define their own class not directly in 'Keras', 
        'XGBoost', or 'sklearn'.
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model. See online documentation for options.
    model_params: dict
        These are hyperparameters required for the construction of the models
        specified by 'major_model' and 'specific_model'. Details on the requirements
        for each model submodel can be found in the online documentation.
    training_params: dict
        These are parameters required for training the models specified by 
        'major_model' and 'specific_model'. Details on the requirements for each
        submodel can be found in the online documentation.
    eval_metric: func: default = mean squared error
        The function used for evaluating cross-validation error. This metric will
        be used to rank model architectures from best to worst. The function must
        take the form 'function(real_values, predicted_values)'.
    custom_model: class: default = None
        A user-defined class passed in when both 'major_model' and 'specific_model'
        are defined as 'Custom'. This model is designed to give greater flexibility
        to users of MLDE. Specific requirements for this custom class can be
        found on the online documentation.
    custom_model_args: iterable: default = []
        Iterable of arguments to pass in to 'custom_model'.
    custom_model_kwargs: dict: default = {}
        Dictionary of kwargs to pass in to 'custom_model'.
    
    Functions
    ---------
    self.train_cv(x, y, train_test_inds)
        Using the base model architecture given by 'self.major_model' and 
        'self.specific_model' trains MldeModel over a set of x and y values with
        k-fold cross validation. All models trained during cross validation are
        stored for use in generating predictions.
    self.predict(self, x):
        Returns the average predicted values for x over all models trained during
        train_cv. 
    self.clear_submodels():
        Force deletion of all trained models, and reset the Keras session. Keras
        sessions are not deleted unless this function is called. Trained models
        are deleted each time self.train_cv() is called.
        
    Attributes
    ----------
    self.major_model
    self.specific_model
    self.model_params
    self.training_params
    """
    # Initialize by defining the major and specific model type
    def __init__(self, major_model, specific_model, model_params = {}, 
                 training_params = {}, eval_metric = mse, 
                 custom_model = None, custom_model_args = [],
                 custom_model_kwargs = {}):
        """
        Stores all input variables as instance variables. 'model_params' and 
        'training_params' are deep-copied prior to being stored as instance
        variables.
        """
        # Store inputs as instance variables
        self._major_model = major_model
        self._specific_model = specific_model
        self._model_params = deepcopy(model_params)
        self._training_params = deepcopy(training_params)
        self._eval_metric = eval_metric
        self._custom_model = custom_model
        self._custom_model_args = custom_model_args
        self._custom_model_kwargs = custom_model_kwargs
        
    # Define a function which builds a model according to the appropriate flavor
    def _build_model(self):
        """
        Private method which builds and returns the model specified by 
        'self.major_model' and 'self.specific_model', using the parameters given
        by 'self._model_params' and 'self._training_params'
        """
        # Confirm that the requested model is real
        assert self._major_model in _class_method_dict, f"Unknown major model: {self._major_model}"
        assert self._specific_model in _class_method_dict[self._major_model],\
            f"Unknown model: {self._major_model}-{self._specific_model}"
            
        # Return a generic model if that's what's requested
        if self._major_model == "Custom" and self._specific_model == "Custom":
            return self._custom_model(*self._custom_model_args, **self._custom_model_kwargs)
            
        # Construct and return the active model
        built_mod = _class_method_dict[self._major_model][self._specific_model]\
            (self._model_params, self._training_params)
        return built_mod
    
    # Define a function for training over a number of cross validation rounds
    def train_cv(self, x, y, train_test_inds, _debug = False):
        """
        Using the base model architecture given by 'self.major_model' and 
        'self.specific_model' trains MldeModel over a set of x and y values with
        k-fold cross validation. All models trained during cross validation are
        stored for use in generating predictions.
        
        Parameters
        ----------
        x: 2D or 3D numpy array, shape N x CL or N x C x L, respectively
            Array containing the encodings of the 'N' amino acid combinations to use
            for training. For all base models other than convolutional neural
            networks, input shape is N x CL (where 'C' is the number of amino acid
            positions bounding the combinatorial space and 'L' is the number of
            latent dimensions in the encoding). Convolutional neural networks
            expect a 3D array.
        y: 1D numpy arrayarray, length N
            Fitness values associated with each x
        train_test_inds: list of lists
            The cross-validation indices use in training.
        
        Returns
        -------
        training_loss: float
            Mean of cross-validation training error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.        
        testing_loss: float
            Mean of cross-validation testing error. Error is calculated using
            the function given by 'eval_metric' upon instantiation.
        """
        # Make sure that x and y are the same length
        assert len(x) == len(y), "Different number of labels and training points"
        
        # Identify cross validation indices and determine the number of models
        # needed.
        assert isinstance(train_test_inds, list), "Expected cross-validation indices to be a list"
        
        # Run checks on the cross-validation indices
        n_models_needed = len(train_test_inds)
        for train_inds, test_inds in train_test_inds:
            unique_train_inds = set(train_inds)
            unique_test_inds = set(test_inds)
            assert len(train_inds) + len(test_inds) == len(x), "Cross val ind error"
            assert len(unique_train_inds) + len(unique_test_inds) == len(x), "Duplicate cross val inds identified"
        
        # Initialize an instance variable for storing models
        self._models = [None for _ in range(n_models_needed)]

        # Initialize an array for storing loss
        training_loss = np.empty(n_models_needed)
        testing_loss = np.empty(n_models_needed)
    
        # Loop over train-test inds
        for i, (train_inds, test_inds) in enumerate(train_test_inds):

            # Build x_train, x_test, y_train, and y_test
            x_train, y_train = x[train_inds], y[train_inds]
            x_test, y_test = x[test_inds], y[test_inds]

            # Build a model
            active_mod = self._build_model()

            # Train the model
            active_mod.train(x_train, y_train, x_test, y_test)
            
            # Get predictions on training and testing data
            train_pred = active_mod.predict(x_train)
            test_pred = active_mod.predict(x_test)
   
            # Record the training and testing error
            training_loss[i] = self._eval_metric(y_train, train_pred)
            testing_loss[i] = self._eval_metric(y_test, test_pred)

            # Save the model object
            self._models[i] = active_mod
        
        # Return details if debug is true
        if _debug:
            return train_test_inds, training_loss, testing_loss
        else:
             # Return mean training and testing losses
            return training_loss.mean(), testing_loss.mean()

    # Write a function for predicting over a number of cross validation rounds
    # and (potentially) zero insertions
    def predict(self, x):
        """
        Returns the average predicted values for x over all models trained during
        train_cv. 
        
        Parameters
        ----------
        x: 2D or 3D numpy array, shape N x CL or N x C x L, respectively
            Array containing the encodings of the 'N' amino acid combinations to use
            for prediction. For all base models other than convolutional neural
            networks, input shape is N x CL (where 'C' is the number of amino acid
            positions bounding the combinatorial space and 'L' is the number of
            latent dimensions in the encoding). Convolutional neural networks
            expect a 3D array.
        
        Returns
        -------
        mean_preds: 1D numpy array: 
            The mean predicted labels over each model generated during training.
        stdev_preds: 1D numpy array:
            The standard deviation of predictions over each model generated
            during training.
        """
        # Create an array to store predictions in. Add an extra dimension if this
        predictions = np.empty([len(self._models), len(x)])
        
        # Loop over the cross-validation models
        for i, model in enumerate(self._models):

            # Make and store predictions
            predictions[i] = model.predict(x).flatten()
            
        # Get the mean and standard deviation of predictions
        mean_preds = np.mean(predictions, axis = 0)
        stdev_preds = np.std(predictions, axis = 0)

        # Return the mean predictions and standard deviation of predictions
        return mean_preds, stdev_preds

    # Write a function that clears all cached models session
    def clear_submodels(self):
        """
        Force deletion of all trained models, and reset the Keras session. Keras
        sessions are not deleted unless this function is called. Trained models
        are deleted each time self.train_cv() is called.
        """
        # If a keras model, clear it
        if self._major_model == "Keras":
            
            # If we have the attribute "_models", delete all
            for model in self._models:
                model.clear_model()
                                                   
        # Delete all active models
        for model in self._models:
            del(model)
            
        # Delete the model and saved_model lists
        del(self._models)
        
        # Collect garbage
        _ = gc.collect()            

    # Set properties of all models people might want to call
    @property
    def major_model(self):
        return self._major_model
    
    @property
    def specific_model(self):
        return self._specific_model
    
    @property
    def model_params(self):
        return self._model_params
    
    @property
    def training_params(self):
        return self._training_params

# Write a class for keras models
class KerasModel():
    """
    Container for all models built in Keras. A number of models are already
    attached to this class, including fully connected neural networks with 
    0, 1, and 2 hidden layers (KerasModel.NoHidden, KerasModel.OneHidden,
    and KerasModel.TwoHidden, respectively) as well as convolutional neural
    networks with 1 and 2 convolutional layers (KerasModel.OneConv, 
    KerasModel.TwoConv). Custom keras models can also be constructed by bypassing
    the connected class methods and passing calling KerasModel directly.
    
    Parameters
    ----------
    mod: uncompiled keras model
        A fully constructed keras model that has not yet been compiled. Compilation
        of the model occurs when KerasModel.train() is called.
    training_params: dict
        Kwargs needed for training the Keras model. "Patience" is a required
        kwarg, and dictates how many epochs will be performed without improvement
        in cross-validation testing error. All other kwargs are passed directly
        to the Model.fit() function in Keras.
        
    Functions
    ---------
    self.train(x_train, y_train, x_test, y_test)
        Trains the input KerasModel with early stopping, using the keyword 
        arguments found in self.training_params. Early stopping is employed.
    self.predict(x):
        Returns predictions on 'x' using the model trained in self.train().
    self.clear_model():
        Clears the Keras session and deletes the trained model.
    KerasModel.NoHidden(model_params, training_params):
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with no hidden layers as the base model. 
    KerasModel.OneHidden(model_params, training_params)
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with one hidden layer as the base model. 
    KerasModel.TwoHidden(model_params, training_params):
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with two hidden layers as the base model. 
    KerasModel.OneConv(model_params, training_params):
        Class method which constructs an instance of KerasModel using a convolutional 
        neural network with one 1D convolutional layer as the base model. 
    KerasModel.TwoConv(model_params, training_params):
        Class method which constructs an instance of KerasModel using a convolutional 
        neural network with two 1D convolutional layers as the base model. 
    
    Attributes
    ----------
    self.early_stopping_epoch
    self.mod
    self.training_params
    """
    # Define an initilization function which sets up all model parameters
    def __init__(self, mod, training_params):
        """
        Assigns the model and training params as instance variables.
        """
        # Set the model as an instance variable
        self._mod = mod

        # Get a local variable of training params
        self._training_params = deepcopy(training_params)
        
        # Compile the model
        self._mod.compile(loss = "mse", optimizer = "adam")
        
        # Set a flag for whether or not the model has been trained
        self._training_performed = False
                
    # Define a function for training the model on one set of x and y
    def train(self, x_train, y_train, x_test, y_test):
        """
        Trains the input KerasModel with early stopping, using the keyword 
        arguments found in self.training_params. Early stopping is employed.
        
        Parameters
        ----------
        x_train: 2D or 3D numpy array, shape N x CL or N x C x L, respectively
            Array containing the encodings of the 'N' amino acid combinations to use
            for training. For fully-connected networks, input shape is N x CL
            (where 'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding). Convolutional neural networks expect a 3D array.
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error. 
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Assert that "patience" is in training params
        assert "patience" in self._training_params, "'patience' kwarg not provided to Keras model training_params"
        
        # Check x and y inputs
        check_training_inputs(x_train, y_train, x_test, y_test)
        
        # If "train" has already been called, throw an error. Otherwise, set that
        # training has been performed
        assert not self._training_performed, "Successive calls to 'train' not supported"
        self._training_performed = True            
        
        # Define callbacks
        callbacks_list = [EarlyStopping(monitor='val_loss',
                                        patience = self._training_params["patience"],
                                        restore_best_weights = True)]
        
        # Build training params without patience
        no_patience_params = {key: val for key, val in self._training_params.items()
                              if key != "patience"}
        
        # Create validation data
        validation_data = (x_test, y_test)
        
        # Fit the model
        _ = self._mod.fit(x_train, y_train, validation_data = validation_data,
                          callbacks = callbacks_list, verbose = 0, **no_patience_params)
                                
        # Identify the early stopping epoch and assign as an attribute
        self._early_stopping_epoch = callbacks_list[0].stopped_epoch
                    
    # Define a function for predicting from a single model instance
    def predict(self, x):
        """
        Returns predictions on 'x' using the model trained in self.train().
        
        Parameters
        ----------
        x: 2D or 3D numpy array, shape N x CL or N x C x L, respectively
            Array containing the encodings of the 'N' amino acid combinations to use
            for training. For fully-connected networks, input shape is N x CL
            (where 'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding). Convolutional neural networks expect a 3D array.
        
        Returns
        -------
        preds: 1D numpy array
            Predicted labels of 'x'.
        """
        # Predict from the model
        preds = self._mod.predict(x).flatten()

        # Return predictions
        return preds

    # Define a function for clearing the model
    def clear_model(self):
        """
        Clears the Keras session and deletes the trained model.
        """
        # Clear the model
        clear_session()
        del(self._mod)
        _ = gc.collect()

    # Create properties
    @property
    def early_stopping_epoch(self):
        return self._early_stopping_epoch
    
    @property
    def mod(self):
        return self._mod
    
    @property
    def training_params(self):
        return self._training_params
    
    # Define a class method that makes a model with no hidden layers
    @classmethod
    def NoHidden(cls, model_params, training_params):
        """
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with no hidden layers as the base model. 
        
        Parameters
        ----------
        model_params: dict
            Must contain 'input_shape' and 'dropout' as kwargs. 'input_shape'
            tells the model the shape of the data to expect (see Keras docs for 
            more detail) and dropout gives the dropout percentage before the
            output layer
        training_params: dict
            Passed directly into KerasModel
            
        Returns
        -------
        KerasModel using a fully-connected neural network with no hidden layers
        as the base model.
        """
        # Check to make sure all expected parameters are present
        check_keras_model_params(("input_shape", "dropout"), 
                                 model_params, "KerasModel.NoHidden")
                
        # Unpack model parameters
        input_shape = model_params["input_shape"]
        dropout = model_params["dropout"]
        
        # Assert that the input shape is 1d
        assert len(input_shape) == 1, "Input shape should be 1d"
        
        # Assert that dropout is between 0 and 1
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0 and 1"

        # Define the model
        model = Sequential()

        # Add a dropout layer followed by a single fully connected layer
        model.add(Dropout(rate = dropout, input_shape = input_shape))
        model.add(Dense(1, activation = "elu"))

        # Build the instance
        return cls(model, training_params)

    # Define a class method that makes a model with a single hidden layer
    @classmethod
    def OneHidden(cls, model_params, training_params):
        """
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with one hidden layer as the base model. 
        
        Parameters
        ----------
        model_params: dict
            Must contain 'input_shape', 'dropout', and 'size' as kwargs. 
            'input_shape' tells the model the same of the data to expect 
            (see Keras docs for more detail), dropout gives the dropout percentage
            before the output layer, and 'size' gives the number of neurons in
            the single hidden layer
        training_params: dict
            Passed directly into KerasModel
            
        Returns
        -------
        KerasModel using a fully-connected neural network with one hidden layer
        as the base model.
        """
        # Check to make sure all expected parameters are present
        check_keras_model_params(("input_shape", "dropout", "size1"),
                                 model_params, "KerasModel.OneHidden")
                
        # Unpack model parameters
        input_shape = model_params["input_shape"]
        dropout = model_params["dropout"]
        size = model_params["size1"]

        # Assert that the input shape is 1d
        assert len(input_shape) == 1, "Input shape should be 1d"

        # Assert that dropout is between 0 and 1
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0 and 1"
        
        # Assert that size params are integers
        assert isinstance(size, int), "Layer size must be an integer"

        # Define the model
        model = Sequential()

        # Add a single hidden layer
        model.add(Dense(int(size), input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Add a dropout layer followed by a single fully connected layer
        model.add(Dropout(rate = dropout))
        model.add(Dense(1, activation = "elu"))

        # Build the instance
        return cls(model, training_params)

    # Define a class method that makes a model with a two hidden layers
    @classmethod
    def TwoHidden(cls, model_params, training_params):
        """
        Class method which constructs an instance of KerasModel using a fully-
        connected neural network with two hidden layers as the base model. 
        
        Parameters
        ----------
        model_params: dict
            Must contain 'input_shape', 'dropout', and 'size1', and 'size1' as kwargs. 
            'input_shape' tells the model the same of the data to expect 
            (see Keras docs for more detail), dropout gives the dropout percentage
            before the output layer, and 'size1' gives the number of neurons in
            the first hidden layer, and 'size2' gives the number of neurons in the
            second hidden layer.
        training_params: dict
            Passed directly into KerasModel
            
        Returns
        -------
        KerasModel using a fully-connected neural network with two hidden layers
        as the base model.
        """
        # Check to make sure all expected parameters are present
        check_keras_model_params(("input_shape", "dropout", "size1", "size2"),
                                 model_params, "KerasModel.TwoHidden")
        
        # Unpack model parameters
        input_shape = model_params["input_shape"]
        dropout = model_params["dropout"]
        size1 = model_params["size1"]
        size2 = model_params["size2"]
    
        # Assert that the input shape is 1d
        assert len(input_shape) == 1, "Input shape should be 1d"
        
        # Assert that dropout is between 0 and 1
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0 and 1"
        
        # Assert that size params are integers
        assert isinstance(size1, int), "Layer size must be an integer"
        assert isinstance(size2, int), "Layer size must be an integer"
    
        # Define the model
        model = Sequential()

        # Add the first hidden layer
        model.add(Dense(int(size1), input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Add the second hidden layer
        model.add(Dense(int(size2)))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Add a dropout layer followed by a single fully connected layer
        model.add(Dropout(rate = dropout))
        model.add(Dense(1, activation = "elu"))

        # Build the instance
        return cls(model, training_params)

    # Define a class method that makes a model with one convolutional layer
    @classmethod
    def OneConv(cls, model_params, training_params):
        """
        Class method which constructs an instance of KerasModel using a convolutional 
        neural network with one 1D convolutional layer as the base model. 
        
        Parameters
        ----------
        model_params: dict
            Must contain 'input_shape', 'dropout', 'filter_choice',
            'n_filters1', and 'flatten_choice' as kwargs. 'input_shape' tells
            the model the same of the data to expect (see Keras docs for more
            detail), dropout gives the dropout percentage before the output
            layer, 'filter_choice' gives the width of the filter, 'n_filters1'
            gives the number of filters, and 'flatten_choice' gives the pooling
            method ('Flatten', 'Max', or 'Average').
        training_params: dict
            Passed directly into KerasModel
            
        Returns
        -------
        KerasModel using a convolutional neural network with one 1D convolutional 
        layer as the base model.
        """
        # Check to make sure all expected parameters are present
        check_keras_model_params(("input_shape", "dropout", "filter_choice", 
                                  "n_filters1", "flatten_choice"),
                                 model_params, "KerasModel.OneConv")
        
        # Unpack model parameters
        input_shape = model_params["input_shape"]
        dropout = model_params["dropout"]
        filter_height = model_params["filter_choice"]
        n_filters = model_params["n_filters1"]
        flatten_choice = model_params["flatten_choice"]
        
        # Assert that the input shape is 2d
        assert len(input_shape) == 2, "Input shape should be 2d"
        
        # Assert that dropout is between 0 and 1
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0 and 1"
        
        # Assert that size params are integers
        assert isinstance(filter_height, int), "filter_choice must be an integer"
        assert isinstance(n_filters, int), "n_filters must be an integer"
        
        # Define the model
        model = Sequential()

        # Add a convolutional layer
        model.add(Conv1D(int(n_filters), filter_height,
                         input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Decide how we enforce going to a single layer
        if flatten_choice=="Flatten":
            model.add(Flatten())
        elif flatten_choice=="Max":
            model.add(GlobalMaxPooling1D())
        elif flatten_choice=="Average":
            model.add(GlobalAveragePooling1D())
        else:
            raise AssertionError("Unexpected flattening choice")
            
        # Add a dropout layer followed by a single fully connected layer
        model.add(Dropout(rate = dropout))
        model.add(Dense(1, activation = "elu"))

        # Build the instance
        return cls(model, training_params)

    # Define a class method that makes a model with two convolutional layers
    @classmethod
    def TwoConv(cls, model_params, training_params):
        """
        Class method which constructs an instance of KerasModel using a convolutional 
        neural network with two 1D convolutional layers as the base model. 
        
        Parameters
        ----------
        model_params: dict
            Must contain 'input_shape', 'dropout', 'filter_arch',
            'n_filters1', 'n_filters2' and 'flatten_choice' as kwargs. 'input_shape'
            tells the model the same of the data to expect (see Keras docs for more
            detail), dropout gives the dropout percentage before the output
            layer, 'filter_arch' is a 2-member iterable giving the width of
            the first and second filters, 'n_filters1' gives the number of
            filters in the first convolutional layer, 'n_filters2' gives the 
            number of filters in the second convolutional layer, and 'flatten_choice'
            gives the pooling method ('Flatten', 'Max', or 'Average').
        training_params: dict
            Passed directly into KerasModel
            
        Returns
        -------
        KerasModel using a convolutional neural network with two 1D convolutional 
        layers as the base model.
        """
        # Check to make sure all expected parameters are present
        check_keras_model_params(("input_shape", "dropout", "filter_arch",
                                  "n_filters1", "n_filters2", "flatten_choice"),
                                 model_params, "KerasModel.TwoConv")
        
        # Unpack model parameters
        input_shape = model_params["input_shape"]
        dropout = model_params["dropout"]
        filter_arch1, filter_arch2 = model_params["filter_arch"]
        n_filters1 = model_params["n_filters1"]
        n_filters2 = model_params["n_filters2"]
        flatten_choice = model_params["flatten_choice"]       
        
        # Assert that the input shape is 2d
        assert len(input_shape) == 2, "Input shape should be 2d"
        
        # Assert that dropout is between 0 and 1
        assert 0.0 <= dropout <= 1.0, "Dropout must be between 0 and 1"
        
        # Assert that size params are integers
        assert isinstance(filter_arch1, int), "filter_arch contents must be integers"
        assert isinstance(filter_arch2, int), "filter_arch contents must be integers"
        assert isinstance(n_filters1, int), "filter_choice1 must be an integer"
        assert isinstance(n_filters2, int), "n_filters2 must be an integer"
        
        # Define the model
        model = Sequential()

        # Add the first convolutional layer
        model.add(Conv1D(int(n_filters1), filter_arch1,
                         input_shape = input_shape))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Add the second convolutional layer
        model.add(Conv1D(int(n_filters2), filter_arch2))
        model.add(BatchNormalization())
        model.add(Activation("elu"))

        # Decide how we enforce going to a single layer
        if flatten_choice=="Flatten":
            model.add(Flatten())
        elif flatten_choice=="Max":
            model.add(GlobalMaxPooling1D())
        elif flatten_choice=="Average":
            model.add(GlobalAveragePooling1D())
        else:
            assert False, "Unexpected flattening choice"

        # Add a dropout layer followed by a single fully connected layer
        model.add(Dropout(rate = dropout))
        model.add(Dense(1, activation = "elu"))

        # Build the instance
        return cls(model, training_params)

# Write a class for the XgbModel
class XgbModel():
    """
    Container for all models built in XGBoost. A number of models are already
    attached to this class, including XgbModel.Linear, XgbModel.Tree, 
    XgbModel.LinearTweedie, and XgbModel.TreeTweedie. 
    
    Parameters
    ----------
    model_params: dict
        These are the parameters passed to xgb.train(), and define the architecture
        of the XGBoost model. See the XGBoost docs for more info on the 'param'
        argument passed in to xgb.train()
    training_params: dict
        These are all optional keyword arguments passed in to xgb.train(). 
        
    Functions
    ---------
    self.train(x_train, y_train, x_test, y_test):
        Trains the input XgbModel with early stopping, using the model defined
        by 'model_params' and training keyword found in 'training_params'.
        Early stopping is employed.
    self.predict(x):
        Generates predicted labels for 'x' based on the model trained in self.train()
    XgbModel.Linear(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Standard regression is used for this model. 
    XgbModel.Tree(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Standard regression is used for this model. 
    XgbModel.LinearTweedie(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Tweedie regression is used for this model.
    XgbModel.TreeTweedie(model_params, training_params):
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Tweedie regression is used for this model.        
        
    Attributes
    ----------
    self.early_stopping_epoch
    self.training_params
    self.model_params
    """
    # Define an initilization function which sets up all model parameters
    def __init__(self, model_params, training_params):
        """
        Copies 'model_params' and 'training_params' and stores them as intance
        variables.
        """
        # Set model and training parameters as instance variables
        self._model_params = deepcopy(model_params)
        self._training_params = deepcopy(training_params)

    # Define a function for training the model on one set of x and y
    def train(self, x_train, y_train, x_test, y_test):
        """
        Trains the input XgbModel with early stopping, using the model defined
        by 'model_params' and training keyword found in 'training_params'.
        Early stopping is employed.
        
        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error. 
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Assert that x is 2D
        assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        assert len(x_test.shape) == 2, "x values must be a 2D matrix"
        
        # Make generic checks on input
        check_training_inputs(x_train, y_train, x_test, y_test)
        
        # Build DMatrices
        train_matrix = xgb.DMatrix(x_train, label = y_train)
        test_matrix = xgb.DMatrix(x_test, label = y_test)

        # Create an eval list
        evallist = [(train_matrix, "train"),
                    (test_matrix, "test")]

        # Train the model and store as the "mod" variable
        self._mod = xgb.train(self._model_params, train_matrix,
                              evals = evallist,
                              **self._training_params)

        # Identify the early stopping epoch
        self._early_stopping_epoch = self._mod.best_ntree_limit

    # Define a function for predicting from a single model instance
    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()
        
        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions 
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding
            
        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """      
        # Assert that x is 2d
        assert len(x.shape) == 2, "Expected a 2D input for x"
        
        # Return predicted values (don't use best iteration if linear, because
        # there is no supported way in xgboost to do this currently...)
        if self._model_params["booster"] == "gblinear":
            preds = self._mod.predict(xgb.DMatrix(x))
        else:
            preds = self._mod.predict(xgb.DMatrix(x),
                                     ntree_limit = self._early_stopping_epoch)
                
        # Return predictions
        return preds.flatten()
        
    # Set properties
    @property
    def early_stopping_epoch(self):
        return self._early_stopping_epoch
    
    @property
    def training_params(self):
        return self._training_params
    
    @property
    def model_params(self):
        return self._model_params
    
    @property
    def mod(self):
        return self._mod
    
    # Define a class method for building a linear booster
    @classmethod
    def Linear(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Standard regression is used for this model. 
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a linear XGBoost model. 
            See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base linear model,
        using 'reg:squarederror' as the XGBoost regression objective and 'rmse'
        as the XGBoost eval metric.
        """
        # Build general model parameters
        mod_params = {"booster": "gblinear",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:squarederror",
                      "eval_metric": "rmse"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a tree booster
    @classmethod
    def Tree(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Standard regression is used for this model. 
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a tree XGBoost model. 
            See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base tree model,
        using 'reg:squarederror' as the XGBoost regression objective and 'rmse'
        as the XGBoost eval metric.
        """
        # Set model parameters
        mod_params = {"booster": "gbtree",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:squarederror",
                      "eval_metric": "rmse"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a Tweedie linear booster
    @classmethod
    def LinearTweedie(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        linear model. Tweedie regression is used for this model.
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a linear XGBoost model
            with the tweedie regression objective. See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base linear model,
        using 'reg:tweedie' as the XGBoost regression objective, 
        'tweedie_variance_power' or 1.5, and 'tweedie-nloglik@1.5' as the
        XGBoost eval metric.
        """
        # Build general model parameters
        mod_params = {"booster": "gblinear",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:tweedie",
                      "tweedie_variance_power": 1.5,
                      "eval_metric": "tweedie-nloglik@1.5"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

    # Define a class method for building a Tweedie tree booster
    @classmethod
    def TreeTweedie(cls, model_params, training_params):
        """
        Generates an instance of XgbModel using an XGBoost model with a base
        tree model. Tweedie regression is used for this model.
        
        Parameters
        ----------
        model_params: dict
            Additional parameters to use when building a tree XGBoost model
            with the tweedie regression objective. See XGBoost docs for details.
        training_params: dict
            Passed directly to XgbModel
            
        Returns
        -------
        An instance of XgbModel using an XGBoost model with a base tree model,
        using 'reg:tweedie' as the XGBoost regression objective, 
        'tweedie_variance_power' or 1.5, and 'tweedie-nloglik@1.5' as the
        XGBoost eval metric.
        """
        # Set model parameters
        mod_params = {"booster": "gbtree",
                      "tree_method": "exact",
                      "nthread": 1,
                      "verbosity": 0,
                      "objective": "reg:tweedie",
                      "tweedie_variance_power": 1.5,
                      "eval_metric": "tweedie-nloglik@1.5"}

        # Add specific model parameters
        mod_params.update(model_params)

        # Create an instance
        return cls(mod_params, training_params)

# Write a class for the SklearnRegressor
class SklearnRegressor():
    """
    Container for sklearn models. 
    
    Parameters
    ----------
    mod: sklearn regressor object
        A regressor object from the scikit-learn machine learning module
    placeholder: dummy variable: default = None
        This variable is not used. It is in place solely to keep this container
        compatible with KerasModel and XgbModel
        
    Functions
    ---------
    self.train(x_train, y_train, x_test, y_test):
        Trains the input sklearn model
    self.predict(x):
        Generates predicted labels for 'x' based on the model trained in self.train()
    SklearnRegressor.Linear(model_params):
        Generates a SklearnRegressor instance using the LinearRegression sklearn
        model.
    SklearnRegressor.GradientBoostingRegressor(model_params):
        Generates a SklearnRegressor instance using the GradientBoostingRegressor sklearn
        model.
    SklearnRegressor.RandomForestRegressor(model_params):
        Generates a SklearnRegressor instance using the RandomForestRegressor sklearn
        model.
    SklearnRegressor.LinearSVR(model_params):
        Generates a SklearnRegressor instance using the LinearSVR sklearn
        model.
    SklearnRegressor.ARDRegression(model_params):
        Generates a SklearnRegressor instance using the ARDRegression sklearn
        model.
    SklearnRegressor.KernelRidge(model_params):
        Generates a SklearnRegressor instance using the KernelRidge sklearn
        model.
    SklearnRegressor.BayesianRidge(model_params):
        Generates a SklearnRegressor instance using the BayesianRidge sklearn
        model.
    SklearnRegressor.BaggingRegressor(model_params):
        Generates a SklearnRegressor instance using the BaggingRegressor sklearn
        model.
    SklearnRegressor.LassoLarsCV(model_params):
        Generates a SklearnRegressor instance using the LassoLarsCV sklearn
        model.
    SklearnRegressor.DecisionTreeRegressor(model_params):
        Generates a SklearnRegressor instance using the DecisionTreeRegressor sklearn
        model.
    SklearnRegressor.SGDRegressor(model_params):
        Generates a SklearnRegressor instance using the SGDRegressor sklearn
        model.
    SklearnRegressor.KNeighborsRegressor(model_params):
        Generates a SklearnRegressor instance using the KNeighborsRegressor sklearn
        model.
    SklearnRegressor.ElasticNet(model_params):
        Generates a SklearnRegressor instance using the ElasticNet sklearn
        model.
    Attributes
    ----------
    self.mod
    """
    # Define an initilization function which sets up all model parameters
    def __init__(self, mod, placeholder = None): 
        """
        Assigns 'mod' as an instance variable.
        """
        # Set the model as an instance variable as well as the model parameters
        self._mod = mod

    # Define a function for training the model on one set of x and y
    def train(self, x_train, y_train, x_test, y_test):
        """
        Trains the input sklearn model
        
        Parameters
        ----------
        x_train: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations to use
            for training.'C' is the number of amino acid positions bounding the
            combinatorial space and 'L' is the number of latent dimensions in
            the encoding
        y_train: 1D numpy array
            Labels to use in training
        x_test: 2D or 3D numpy array
            x-values to use in calcuating test error. 
        y_test: 1D numpy array
            Labels to use in calculating test error.
        """
        # Test the input data
        check_training_inputs(x_train, y_train, x_test, y_test)
        
        # Make sure x is 2D
        assert len(x_train.shape) == 2, "x values must be a 2D matrix"
        assert len(x_test.shape) == 2, "x values must be a 2D matrix"
        
        # If the model is LinearRegressor, reshape y_train to be 2d
        if self._mod.__class__.__name__ == "LinearRegression":
            y_train = np.expand_dims(y_train, axis = 1)

        # Fit the model
        self._mod.fit(x_train, y_train)

    # Define a function for predicting from a single model instance
    def predict(self, x):
        """
        Generates predicted labels for 'x' based on the model trained in self.train()
        
        Parameters
        ----------
        x: 2D numpy array, shape N x CL
            Array containing the encodings of the 'N' amino acid combinations for
            which to predict labels.'C' is the number of amino acid positions 
            bounding the combinatorial space and 'L' is the number of latent
            dimensions in the encoding
            
        Returns
        -------
        preds: 1D numpy array
            Predicted labels for 'x'
        """      
        # Throw an error if x is not 2D
        assert len(x.shape) == 2, "x must be 2D"    
    
        # Return the prediction
        return self._mod.predict(x).flatten()
    
    # Create properties
    @property
    def mod(self):
        return self._mod

    # Define a class method for building a linear regressor
    @classmethod
    def Linear(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LinearRegression sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LinearRegression class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn model instance
        mod = LinearRegression(**model_params)

        # Construct with the initializer
        return cls(mod)

    # Define a class method for building with a gradient boosting regressor
    @classmethod
    def GradientBoostingRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the GradientBoostingRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's GradientBoostingRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = GradientBoostingRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for building with a RandomForestRegressor
    @classmethod
    def RandomForestRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the RandomForestRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's RandomForestRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = RandomForestRegressor(**model_params)

        # Create an instance
        return cls(mod)

    # Define a class method for building LinearSVR
    @classmethod
    def LinearSVR(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LinearSVR sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LinearSVR class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = LinearSVR(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for ARDRegression
    @classmethod
    def ARDRegression(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the ARDRegression sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's ARDRegression class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = ARDRegression(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for KernelRidge
    @classmethod
    def KernelRidge(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the KernelRidge sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's KernelRidge class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = KernelRidge(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for BayesianRidge
    @classmethod
    def BayesianRidge(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the BayesianRidge sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's BayesianRidge class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = BayesianRidge(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for BaggingRegressor
    @classmethod
    def BaggingRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the BaggingRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's BaggingRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = BaggingRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a class method for LassoLarsCV
    @classmethod
    def LassoLarsCV(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the LassoLarsCV sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's LassoLarsCV class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = LassoLarsCV(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for DecisionTreeRegressor
    @classmethod
    def DecisionTreeRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the DecisionTreeRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's DecisionTreeRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = DecisionTreeRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for SGDRegressor
    @classmethod
    def SGDRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the SGDRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's SGDRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = SGDRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for KNeighborsRegressor
    @classmethod
    def KNeighborsRegressor(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the KNeighborsRegressor sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's KNeighborsRegressor class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = KNeighborsRegressor(**model_params)

        # Return an instance
        return cls(mod)

    # Define a classmethod for ElasticNet
    @classmethod
    def ElasticNet(cls, model_params, training_params = None):
        """
        Generates a SklearnRegressor instance using the ElasticNet sklearn
        model.
        
        Parameters
        ----------
        model_params: dict
            Kwargs passed in to sklearn's ElasticNet class
        training_params: None
            Unused. Present to remain compatible with other containers.
        """
        # Build the sklearn instance
        mod = ElasticNet(**model_params)

        # Return an instance
        return cls(mod)
        
# Define a dictionary structure for calling class methods
_class_method_dict = {"Keras": {"NoHidden": KerasModel.NoHidden,
                               "OneHidden": KerasModel.OneHidden,
                               "TwoHidden": KerasModel.TwoHidden,
                               "OneConv": KerasModel.OneConv,
                               "TwoConv": KerasModel.TwoConv,
                               "Custom": KerasModel},
                     "XGB": {"Tree": XgbModel.Tree,
                             "Linear": XgbModel.Linear,
                             "Tree-Tweedie": XgbModel.TreeTweedie,
                             "Linear-Tweedie": XgbModel.LinearTweedie,
                             "Custom": XgbModel},
                     "sklearn-regressor": {"Linear": SklearnRegressor.Linear,
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
                                           "ElasticNet": SklearnRegressor.ElasticNet,
                                           "Custom": SklearnRegressor},
                     "Custom": {"Custom"}
}