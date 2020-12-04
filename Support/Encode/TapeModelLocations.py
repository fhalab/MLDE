# Import required modules
import os

# Get the directory of this file
_filedir = os.path.dirname(os.path.abspath(__file__))

# Define the base weight location
_base_weight_loc = os.path.join(_filedir, "..", "tape-neurips2019", "pretrained_models")

# Define the location of pretrained weights for each model in the TAPE package
tape_model_locations = {"resnet": os.path.join(_base_weight_loc, "resnet_weights.h5"),
                        "bepler": os.path.join(_base_weight_loc, "bepler_unsupervised_pretrain_weights.h5"),
                        "unirep": os.path.join(_base_weight_loc, "unirep_weights.h5"),
                        "transformer": os.path.join(_base_weight_loc, "transformer_weights.h5"),
                        "lstm": os.path.join(_base_weight_loc, "lstm_weights.h5")}