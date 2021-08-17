"""
This file contains all functions needed for running DeepSequence. It is 
essentially a combination of `run_svi.py` and examples from 
`Mutation Effect Prediction.ipynb` in the original DeepSequence repo (with
appropriate modificaitons to allow generalizability).
"""
# Load third party modules
import itertools
import os
import pandas as pd

# Load DeepSequence objects
from DeepSequence import helper, model, train
from .globals import (MODEL_PARAMS, TRAIN_PARAMS, DEEPSEQ_WORKING_DIR,
                      ALL_AAS, N_PRED_ITERATIONS)

# Write a function that trains the model
def train_model(alignment_file, save_model_params = False):
    """
    Trains a VAE using DeepSequence. This function is essentially the contents
    of `run_svi.py` on the DeepSequence GitHub repo. Note that all arguments to
    training DeepSequence are input as global variables. The globals are all
    defined in `globals.py` in this same directory.
    
    Parameters
    ----------
    alignment_file: str: Path to the .a2m file to use for training DeepSequence.
        The first sequence in the file should be the reference sequence.
    save_model_params: bool. Default = False: Whether or not to save the model
        parameters after training. If saved, model parameters will be saved to
        the DeepSequence submodule folder 'examples/params/'.
        
    Returns
    -------
    data_helper: DataHelper obj from DeepSequence: A wrapper containing all 
        information about the input alignment file.
    vae_model: VariationalAutoencoder obj from DeepSequence: A trained VAE to
        be used for making zero-shot predictions.
    """
    # Build the data helper. We modify this relative to the run_svi.py 
    # script on the DeepSequence repo so that we can pass in an alignment
    # file rather than a dataset
    data_helper = helper.DataHelper(alignment_file = alignment_file,
                                    working_dir = DEEPSEQ_WORKING_DIR,
                                    calc_weights=True)

    # Build the model. This is straight from run_svi.py, but the 
    # `model_params` variable has been replaced with our global
    vae_model   = model.VariationalAutoencoder(data_helper,
        batch_size                     =   MODEL_PARAMS["bs"],
        encoder_architecture           =   [MODEL_PARAMS["encode_dim_zero"],
                                                MODEL_PARAMS["encode_dim_one"]],
        decoder_architecture           =   [MODEL_PARAMS["decode_dim_zero"],
                                                MODEL_PARAMS["decode_dim_one"]],
        n_latent                       =   MODEL_PARAMS["n_latent"],
        logit_p                        =   MODEL_PARAMS["logit_p"],
        sparsity                       =   MODEL_PARAMS["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   MODEL_PARAMS["final_decode_nonlin"],
        final_pwm_scale                =   MODEL_PARAMS["final_pwm_scale"],
        conv_decoder_size              =   MODEL_PARAMS["d_c_size"],
        convolve_patterns              =   MODEL_PARAMS["conv_pat"],
        n_patterns                     =   MODEL_PARAMS["n_pat"],
        random_seed                    =   MODEL_PARAMS["r_seed"],
        )

    # Build the job string. 
    job_string = helper.gen_job_string({"MLDE": "MLDE"}, MODEL_PARAMS)

    # Train the model. Again, this is straight form run_svi.py, but the
    # `train_params` variable has been replaced with our global
    train.train(data_helper, vae_model,
        num_updates             =   TRAIN_PARAMS["num_updates"],
        save_progress           =   TRAIN_PARAMS["save_progress"],
        save_parameters         =   TRAIN_PARAMS["save_parameters"],
        verbose                 =   TRAIN_PARAMS["verbose"],
        job_string              =   job_string)

    # Save the model if requested
    if save_model_params:
        vae_model.save_parameters(file_prefix=job_string)

    return data_helper, vae_model

def build_mutant_file(input_positions):
    """
    DeepSequence can make predictions in batch if we provide it a csv file 
    containing mutations. This function builds such a file for a combinatorial
    library. This csv file will be saved in the DeepSequence submodule at
    'examples/mutations/MldeMuts.csv' and will be overwritten for each run. 
    
    Parameters
    ----------
    input_positions: iterable of str: The iterable should return amino acid 
        positions to include in the combinatorial library. Each position is 
        formatted as AA#. 
        
    Returns
    -------
    mutfile_loc_rel_to_working_dir: str: This string gives the location of the
        output file relative to the DeepSequence working directory (which is
        'DeepSequence/examples' by default). DeepSequence assumes that some files
        are at specific locations within its directory structure, so the location
        of the mutation file must be given assuming we are operating from within
        the DeepSequence submodule.
    detail_to_shorthand: dict: Converts long-form amino acid combination names 
        (which have the format AA1#AA2:AA1#AA2...) to short-form amino acid
        combination names (just the mutant amino acid).
    """
    # Get the number of positions and build a combinatorial library
    n_positions = len(input_positions)
    all_mutants = []
    detail_to_shorthand = {}
    for i, combo in enumerate(itertools.product(ALL_AAS, repeat = n_positions)):

        # Build the mutation list
        mutation_list = ["".join([input_positions[j], mutant_char])
                        for j, mutant_char in enumerate(combo)
                        if mutant_char != input_positions[j][0]]

        # Continue if there is are no elements in the mutation list (WT)
        if len(mutation_list) == 0:
            continue

        # Build the mutants
        detailed_mutant = ":".join(mutation_list)
        all_mutants.append(detailed_mutant)
        detail_to_shorthand[detailed_mutant] = "".join(combo)

    # Make sure we have the correct number of mutants
    assert len(all_mutants) == (20**n_positions - 1)
    assert len(all_mutants) == len(detail_to_shorthand)
    
    # Save to a buffer location
    mutant_df = pd.DataFrame({"mutant": all_mutants})
    mutfile_loc_rel_to_working_dir = os.path.join("mutations", "MldeMuts.csv")
    mutfile_loc = os.path.join(DEEPSEQ_WORKING_DIR, mutfile_loc_rel_to_working_dir)
    mutant_df.to_csv(mutfile_loc, index = False)
    
    return mutfile_loc_rel_to_working_dir, detail_to_shorthand

def predict_mutants(data_helper, vae_model, input_positions, mutfile_loc,
                    detail_to_shorthand, _include_assert = True, _iter_override = None):
    """
    Uses a trained DeepSequence VAE model to get delta-elbos for a combinatorial
    library at a set of positions.
    
    Parameters
    ----------
    data_helper: DataHelper obj from DeepSequence: A wrapper containing all 
        information about the input alignment file.
    vae_model: VariationalAutoencoder obj from DeepSequence: A trained VAE to
        be used for making zero-shot predictions.
    input_positions: iterable of str: The iterable should return amino acid 
        positions to include in the combinatorial library. Each position is 
        formatted as AA#.
    mutfile_loc: str: This string gives the location of the output file relative
        to the DeepSequence working directory (which is 'DeepSequence/examples'
        by default). DeepSequence assumes that some files are at specific
        locations within its directory structure, so the location of the
        mutation file must be given assuming we are operating from within the
        DeepSequence submodule.
    detail_to_shorthand: dict: Converts long-form amino acid combination names 
        (which have the format AA1#AA2:AA1#AA2...) to short-form amino acid
        combination names (just the mutant amino acid).
        
    Returns
    -------
    deep_seq_results: pd.DataFrame: Dataframe containing the delta-elbos mapped
        to each shorthand combination in the combinatorial library. 
    """
    # Get the number of prediction iterations
    if _iter_override is not None:
        pred_iters = _iter_override
    else:
        pred_iters = N_PRED_ITERATIONS
    
    # Make predictions
    mutant_name_list, delta_elbos = data_helper.custom_mutant_matrix(mutfile_loc, vae_model, 
                                                                     N_pred_iterations=pred_iters)

    # Make sure all mutants have a calculation
    if _include_assert:
        n_mutant_preds = len(mutant_name_list)
        assert len(detail_to_shorthand) == n_mutant_preds, "Missing calculation for DeepSeq. Maybe missing positions in the input alignment?"
        assert n_mutant_preds == (20**len(input_positions) - 1), "Unexpected number of predictions"

    # Map mutant name list to shorthand combos
    shorthand_name_list = [detail_to_shorthand[combo] for combo in mutant_name_list]

    # Create a dataframe of results
    deep_seq_results = pd.DataFrame({"Combo": shorthand_name_list,
                                     "DeepSequence": delta_elbos})

    # Add the wild type sequence and return
    wt_combo = "".join([pos[0] for pos in input_positions])
    return deep_seq_results.append(pd.DataFrame({"Combo": [wt_combo], "DeepSequence": [0]}),
                                   ignore_index = True)

# Now write a wrapper function that includes all steps
def run_deepseq(alignment_loc, target_positions, save_model = False):
    """
    Wraps all of the above functions to perform zero-shot prediction using 
    DeepSequence.
    
    Parameters
    ----------
    alignment_loc: str: Path to the .a2m file to use for training DeepSequence.
        The first sequence in the file should be the reference sequence.
    target_positions: iterable of str: The iterable should return amino acid 
        positions to include in the combinatorial library. Each position is 
        formatted as AA#. 
    save_model: bool. Default = False: Whether or not to save the model
        parameters after training. If saved, model parameters will be saved to
        the DeepSequence submodule folder 'examples/params/'.
        
    Returns
    -------
    deep_seq_results: pd.DataFrame: Dataframe containing the delta-elbos mapped
        to each shorthand combination in the combinatorial library. 
    """
    # Build the file containing all mutants
    mutfile_loc, detail_to_shorthand = build_mutant_file(target_positions)
    
    # Train a DeepSequence model
    data_helper, trained_model = train_model(alignment_loc,
                                             save_model_params = save_model)

    # Make zero-shot predictions
    print ("Making predictions. This can take a bit...")
    return predict_mutants(data_helper, trained_model, target_positions,
                           mutfile_loc, detail_to_shorthand)