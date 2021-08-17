"""
Contains functions that support running zero-shot predictions
"""
# Import 3rd party modules
import os

# Import custom modules
from .zero_shot_predictor import ZeroShotPredictor
from ..encode.model_info import TRANSFORMER_INFO

def check_args(args):
    """
    Confirms a number of assumptions made in the handling of arguments input
    to `zero_shot_predictor.py`. Note that `run_zero_shot` has a number of 
    assertion checks as well. This functions provides checks for the most common
    errors and not for all possible.
    """
    # Get the set of allowed models and the set of requested models
    allowed_models = set(TRANSFORMER_INFO.keys()) | set(["EVmutation"])
    requested_models = set(args.models)
    
    # Make sure there are no duplicate models or positions
    assert len(requested_models) == len(args.models), "Duplicate model inputs"
    assert len(args.positions) == len(set(args.positions)), "Duplicate positions"
    
    # Confirm that at least one model was requested
    assert len(requested_models) > 0, "Did not request a model"
    
    # If EVmutation is requested, then the evmutation_model must be provided
    if "EVmutation" in requested_models:
        assert args.evmutation_model != None, "Must provide `evmutation_model` for EVmutation"
        assert os.path.isfile(args.evmutation_model), "Cannot find `evmutation_model`"
            
    # If the MSA transformer is requested, then the alignment attribute must
    # be provided
    if "esm_msa1_t12_100M_UR50S" in requested_models:
        assert args.alignment != None, "Must provide `alignment` for MSA transformer"
        assert os.path.isfile(args.alignment), "Cannot find `alignment` file"
        
    # If anything other than the MSA transformer is requested, then the fasta 
    # atttribute must be provided
    else:
        assert args.fasta != None, "Must provide `fasta`"
        assert os.path.isfile(args.fasta), "Cannot find `fasta` file"
    
    # Find requested models that are unrecognized. Report unrecognized.
    unrecognized_models = requested_models - allowed_models
    if len(unrecognized_models) > 0:
        print("Models should be one of...")
        for model_name in allowed_models:
            print(model_name)
        raise AssertionError(f"Unrecognized models: {', '.join(unrecognized_models)}")
        
    # Confirm that at least one position was requested
    assert len(args.positions) > 0, "Must specifiy at least one position"
    
    # Make sure that the batch size is a number greater than or equal to 1
    assert args.batch_size >= 1, "Batch size cannot be <1"
    
    # Make sure the requested output location exists
    assert os.path.isdir(args.output), f"Directory does not exist: {args.output}"

def run_zero_shot(args):
    """
    Performs zero-shot prediction using requested models.
    """
    # Define placeholder predictors
    non_msa_predictor = None
    
    # Loop over all models
    all_preds = []
    for model_name in args.models:
        
        # Report what we're working on
        print(f"Making zero-shot predictions using {model_name}...")
        
        # Go down this path if not an msa transformer
        if model_name != "esm_msa1_t12_100M_UR50S":
            
            # If the predictor is None, then build one
            if non_msa_predictor == None:
                non_msa_predictor = ZeroShotPredictor(args.fasta, 
                                                      args.positions,
                                                      msa_transformer = False)
                
            # Predict with EVmutation
            if model_name == "EVmutation":
                all_preds.append(non_msa_predictor.predict_evmutation(args.evmutation_model))
                
            # Otherwise, predict with a transformer
            else:
                
                # Predict using naive probability
                all_preds.append(non_msa_predictor.predict_esm(model_name,
                                                               batch_size = args.batch_size,
                                                               naive = True))
                
                # If including conditional probability, predict here too
                if args.include_conditional:
                    all_preds.append(non_msa_predictor.predict_esm(model_name,
                                                                   batch_size = args.batch_size,
                                                                   naive = False))
        
        # If this is an msa transformer, build and predict with a different
        # loaded fasta file
        else:
            
            # Build predictor
            msa_predictor = ZeroShotPredictor(args.alignment, args.positions,
                                              msa_transformer = True)
            
            # Predict using naive probability
            all_preds.append(msa_predictor.predict_esm(model_name,
                                                       batch_size = args.batch_size,
                                                       naive = True,
                                                       full_col = args.mask_col))
            
            # If including conditional probability, predict here too
            if args.include_conditional:
                all_preds.append(msa_predictor.predict_esm(model_name,
                                                           batch_size = args.batch_size,
                                                           naive = False,
                                                           full_col = args.mask_col))
                
        # At the end of a loop, join all predictions and save
        combined_preds = all_preds[0]
        if len(all_preds) > 1:
            for pred_df in all_preds[1:]:
                combined_preds = combined_preds.merge(pred_df, how = "left",
                                                      on = "Combo")
                
        # Save results
        combined_preds.to_csv(os.path.join(args.output, "ZeroShotPreds.csv"),
                              index = False)