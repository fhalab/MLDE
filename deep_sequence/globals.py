"""
This file contains global parameters for running zero-shot prediction with
DeepSequence. The default model parameters needed for loading and training a
DeepSequence model are taken directly from the DeepSequence git repo file 
"run_svi.py"
"""
import os

# Import the directory holding the DeepSequence code
from . import DEEPSEQ_DIR_PATH

# Define model and training parameters needed to define the default DeepSequence
# model
MODEL_PARAMS = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }
TRAIN_PARAMS = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   False,
    }

# Define the location of the working dir. This is an internal deep sequence
# parameter that keeps track of where to send log files.
DEEPSEQ_WORKING_DIR = os.path.join(DEEPSEQ_DIR_PATH, "examples")

# Set the number of prediction iterations and minibatch size of making predictions
N_PRED_ITERATIONS = 2000

# Define an allowed set of amino acids
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")