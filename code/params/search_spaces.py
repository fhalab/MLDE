"""
This file contains all information relevant to running hyperopt with MLDE. The 
only things that can be safely changed in this file are the ranges and priors
on the search spaces defined in 'search_spaces'
"""
# Import relevant modules
from hyperopt import hp

# Define generic parameters that will be searched over
_nohidden_space = ("dropout",)
_onehidden_space = ("dropout", "size1")
_twohidden_space = ("dropout", "size1", "size2")
_oneconv_space = ("dropout", "filter_choice", "n_filters1", "flatten_choice")
_twoconv_spave = ("dropout", "filter_arch", "n_filters1", 
                 "n_filters2", "flatten_choice")

_xgbtree_space = ("eta", "max_depth", "lambda", "alpha")
_xgblinear_space = ("lambda", "alpha")
                  
_linear_space = ("dummy",)
_gradientboosting_space = ("learning_rate", "n_estimators", "min_samples_split",
                          "min_samples_leaf", "max_depth")
_randomforest_space = ("n_estimators", "min_samples_split", "min_samples_leaf",
                      "max_depth")
_linearsvr_space = ("tol", "C", "dual")
_ardregression_space = ("tol", "alpha_1", "alpha_2", "lambda_1", "lambda_2")
_kernelridge_space = ("alpha", "kernel")
_bayesianridge_space = ("tol", "alpha_1", "alpha_2", "lambda_1", "lambda_2")
_bagging_space = ("n_estimators", "max_samples")
_lassolars_space = ("max_iter", "cv", "max_n_alphas")
_decisiontreeregressor_space = ("max_depth", "min_samples_split",
                               "min_samples_leaf")
_sgdregressor_space = ("alpha", "l1_ratio", "tol")
_kneighborsregressor_space = ("n_neighbors", "weights", "leaf_size", "p")
_elasticnet_space = ("l1_ratio", "alpha")
_adaboost_space = ("n_estimators", "learning_rate")

# Define choice tuples for each time a choice is made in hyperopt
CATEGORICAL_PARAMS = {"filter_choice": (1, 0.75, 0.5, 0.25),
           "filter_arch": ((1, 0.25), (0.75, 0.5), (0.75, 0.25), (0.5, 0.5), (0.5, 0.25)),
           "flatten_choice": ("Flatten", "Max", "Average"),
           "dual": (True, False),
           "weights": ("uniform", "distance"),
           "kernel": ("linear", "rbf", "laplacian", "polynomial",
                      "chi2", "sigmoid")}

# Define a list of values that must be converted to integers after coming from
# hyperopt
INTEGER_PARAMS = {"max_depth", "n_filters1", "n_filters2", "n_estimators", 
                  "n_neighbors", "leaf_size", "max_iter", "cv", "max_n_alphas",
                  "size1", "size2"}

# Define a set of values that are written in the search space as the fraction of
# total latent space
LATENT_PERC_PARAMS = {"size1", "size2", "n_filters1", "n_filters2"}

# Define search spaces for all available parameters
SEARCH_SPACES = {"XGB": {"eta": hp.uniform("eta", 0.01, 0.5),
                  "max_depth": hp.quniform("max_depth", 2, 10, 1),
                  "lambda": hp.uniform("lambda", 0, 10),
                  "alpha": hp.uniform("alpha", 0, 10)},
          "Keras": {"dropout": hp.uniform("dropout", 0, 0.5),
                    "size1": hp.uniform("size1", 0.25, 0.75),
                    "size2": hp.uniform("size2", 1/32, 0.25),
                    "filter_choice": hp.choice("filter_choice", 
                                               CATEGORICAL_PARAMS["filter_choice"]),
                    "filter_arch": hp.choice("filter_arch", CATEGORICAL_PARAMS["filter_arch"]),
                    "n_filters1": hp.uniform("n_filters1", 0.0625, 0.25),
                    "n_filters2": hp.uniform("n_filters2", 1/256, 0.0625),
                    "flatten_choice": hp.choice("flatten_choice", 
                                                CATEGORICAL_PARAMS["flatten_choice"])},
          "sklearn-regressor": {"n_estimators": hp.quniform("n_estimators", 10, 500, 1),
                      "learning_rate": hp.uniform("learning_rate", 0.01, 1),
                      "max_samples": hp.uniform("max_samples", 0.1, 1),
                      "max_depth": hp.quniform("max_depth", 3, 10, 1),
                      "min_samples_split": hp.uniform("min_samples_split", 0.005, 0.03),
                      "min_samples_leaf": hp.uniform("min_samples_leaf", 0.002, 0.03),
                      "tol": hp.uniform("tol", 1e-5, 1e-3),
                      "C": hp.uniform("C", 0.1, 10),
                      "dual": hp.choice("dual", CATEGORICAL_PARAMS["dual"]),
                      "alpha_1": hp.uniform("alpha_1", 1e-7, 1e-5),
                      "alpha_2": hp.uniform("alpha_2", 1e-7, 1e-5),
                      "lambda_1": hp.uniform("lambda_1", 1e-7, 1e-5),
                      "lambda_2": hp.uniform("lambda_2", 1e-7, 1e-5),
                      "n_neighbors": hp.quniform("n_neighbors", 1, 30, 1),
                      "weights": hp.choice("weights", CATEGORICAL_PARAMS["weights"]),
                      "leaf_size": hp.quniform("leaf_size", 1, 50, 1),
                      "p": hp.uniform("p", 1, 2),
                      "l1_ratio": hp.uniform("l1_ratio", 0, 1),
                      "alpha": hp.uniform("alpha", 1e-4, 10),
                      "kernel": hp.choice("kernel", CATEGORICAL_PARAMS["kernel"]),
                      "max_iter": hp.quniform("max_iter", 10, 1000, 1),
                      "cv": hp.quniform("cv", 2, 10, 1),
                      "max_n_alphas": hp.quniform("max_n_alphas", 10, 2000, 1),
                      "dummy": hp.uniform("dummy", 0, 1)}}

# Define the generic search spaces for each model type
SPACE_BY_MODEL = {"Keras": {"NoHidden": _nohidden_space,
                            "OneHidden": _onehidden_space,
                            "TwoHidden": _twohidden_space,
                            "OneConv": _oneconv_space,
                            "TwoConv": _twoconv_spave},
                  "XGB": {"Tree": _xgbtree_space,
                          "Linear": _xgblinear_space,
                          "Tree-Tweedie": _xgbtree_space,
                          "Linear-Tweedie": _xgblinear_space},
                  "sklearn-regressor": {"Linear": _linear_space,
                                        "GradientBoostingRegressor": _gradientboosting_space,
                                        "RandomForestRegressor": _randomforest_space,
                                        "LinearSVR": _linearsvr_space,
                                        "ARDRegression": _ardregression_space,
                                        "KernelRidge": _kernelridge_space,
                                        "BayesianRidge": _bayesianridge_space,
                                        "BaggingRegressor": _bagging_space,
                                        "LassoLarsCV": _lassolars_space,
                                        "DecisionTreeRegressor": _decisiontreeregressor_space,
                                        "SGDRegressor": _sgdregressor_space,
                                        "KNeighborsRegressor": _kneighborsregressor_space,
                                        "ElasticNet": _elasticnet_space,
                                        "AdaBoostRegressor": _adaboost_space}}