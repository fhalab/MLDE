"""
This file contains all default training and model parameters for MLDE. It also
tells the program which models are designed to be run on the GPU and which are 
designed to be run on the CPU. Changing default parameters in here will change
the defaults used. Changing which models are assigned as CPU and GPU will just
break the program. 

Note that all Keras parameters are defined relative to the input shape of the
data (e.g. size parameters are floats between 0 and 1). 
"""
# Define the default model parameters
default_model_params = {"Keras": {"NoHidden": {"dropout": 0.2},
                                 "OneHidden": {"dropout": 0.2,
                                               "size1": 0.25},
                                 "TwoHidden": {"dropout": 0.2,
                                               "size1": 0.25,
                                               "size2": 0.0625},
                                 "OneConv": {"dropout": 0.2,
                                             "filter_choice": 0.5,
                                             "n_filters1": 0.0625,
                                             "flatten_choice": "Average"},
                                 "TwoConv": {"dropout": 0.2,
                                             "filter_arch": (0.5, 0.5),
                                             "n_filters1": 0.0625,
                                             "n_filters2": 1/128,
                                             "flatten_choice": "Average"}
                                 },
                      "XGB": {"Tree": {"eta": 0.3,
                                       "max_depth": 6,
                                       "lambda": 1,
                                       "alpha": 0},
                              "Linear": {"lambda": 1,
                                         "alpha": 0},
                              "Tree-Tweedie": {"eta": 0.3,
                                               "max_depth": 6,
                                               "lambda": 1,
                                               "alpha": 0},
                              "Linear-Tweedie": {"lambda": 1,
                                                 "alpha": 0}
                              },
                      "sklearn-regressor": {"Linear": {},
                                            "GradientBoostingRegressor": {},
                                            "RandomForestRegressor": {"n_estimators": 100},
                                            "BayesianRidge": {},
                                            "LinearSVR": {},
                                            "ARDRegression": {},
                                            "KernelRidge": {},
                                            "BaggingRegressor": {},
                                            "LassoLarsCV": {"cv": 5},
                                            "DecisionTreeRegressor": {},
                                            "SGDRegressor": {},
                                            "KNeighborsRegressor": {},
                                            "ElasticNet": {}
                                            }
                      }

# Define the default training parameters
default_training_params = {"Keras": {"patience": 10, 
                                   "batch_size": 32,
                                   "epochs": 1000},
                         "XGB": {"early_stopping_rounds": 10,
                                 "num_boost_round": 1000,
                                 "verbose_eval": False},
                         "sklearn-regressor": {}
                         }

# Define the CPU models
cpu_models = (("XGB", "Tree"),
              ("XGB", "Linear"),
              ("XGB", "Tree-Tweedie"),
              ("XGB", "Linear-Tweedie"),
              ("sklearn-regressor", "Linear"),
              ("sklearn-regressor", "GradientBoostingRegressor"),
              ("sklearn-regressor", "RandomForestRegressor"),
              ("sklearn-regressor", "BayesianRidge"),
              ("sklearn-regressor", "LinearSVR"),
              ("sklearn-regressor", "ARDRegression"),
              ("sklearn-regressor", "KernelRidge"),
              ("sklearn-regressor", "BaggingRegressor"),
              ("sklearn-regressor", "LassoLarsCV"),
              ("sklearn-regressor", "DecisionTreeRegressor"),
              ("sklearn-regressor", "SGDRegressor"),
              ("sklearn-regressor", "KNeighborsRegressor"),
              ("sklearn-regressor", "ElasticNet")
              )

# Define the GPU models
gpu_models = (("Keras", "NoHidden"),
              ("Keras", "OneHidden"),
              ("Keras", "TwoHidden"),
              ("Keras", "OneConv"),
              ("Keras", "TwoConv")
              )