# Import pytest and functions to be tested
import pytest
from ....zero_shot.support_funcs import check_args

# Import other required modules/objects
from ....encode.model_info import TRANSFORMER_INFO
import os
import string
import random

# Define the models allowed for zero-shot prediction
ALLOWED_MODELS = list(TRANSFORMER_INFO.keys()) + ["EVmutation"]

# Create a set of dummy positions
DUMMY_POSITIONS = ["A", "B"]

# Get a location for an evmutation model, a fasta sequence, and an alignment
FASTA_LOC = "./code/validation/basic_test_data/2GI9.fasta"
ALIGNMENT_LOC = "./code/validation/basic_test_data/GB1_Alignment.a2m"
EVMUT_LOC = "./code/validation/basic_test_data/GB1_EVcouplingsModel.model"

# Build a dummy namespace class to pass in to the check_args function
class DummyNamespace():
    
    def __init__(self, positions, models, fasta = None,
                alignment = None, evmutation_model = None,
                include_conditional = False,
                mask_col = False, batch_size = 4, output = os.getcwd()):
        
        self.positions = positions
        self.models = models
        self.fasta = fasta
        self.alignment = alignment
        self.evmutation_model = evmutation_model
        self.include_conditional = include_conditional
        self.mask_col = mask_col
        self.batch_size = batch_size
        self.output = output        

# Test the checks
def test_check_args():
    
    # Error on duplicate model inputs
    dup_mods = ALLOWED_MODELS + ALLOWED_MODELS
    with pytest.raises(AssertionError, match = "Duplicate model inputs"):
        check_args(DummyNamespace(DUMMY_POSITIONS, dup_mods))
        
    # Error on duplicate positions
    dup_positions = ["A15", "A15", "G16"]
    with pytest.raises(AssertionError, match = "Duplicate positions"):
        check_args(DummyNamespace(dup_positions, ALLOWED_MODELS))
        
    # Error on empty models
    with pytest.raises(AssertionError, match = "Did not request a model"):
        check_args(DummyNamespace(DUMMY_POSITIONS, []))
        
    # Confirm error for evmutation if no model provided
    with pytest.raises(AssertionError, match = "Must provide `evmutation_model` for EVmutation"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS))
        
    # Confirm error for evmutation if evmutation model not provided
    with pytest.raises(AssertionError, match = "Cannot find `evmutation_model`"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS, 
                                evmutation_model = "./FAKE.txt"))
        
    # Confirm error for msa transformer if alignment not provided
    with pytest.raises(AssertionError, match = "Must provide `alignment` for MSA transformer"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS, 
                                evmutation_model = EVMUT_LOC))
        
    # Confirm error for msa transformer if alignment cannot be found
    with pytest.raises(AssertionError, match = "Cannot find `alignment` file"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS, 
                                evmutation_model = EVMUT_LOC,
                                alignment = "./FAKE.txt"))
        
    # Confirm error for all other models if fasta not provided or not found
    for model in ALLOWED_MODELS:
        if model != "esm_msa1_t12_100M_UR50S":
            
            # Error if no fasta provided
            with pytest.raises(AssertionError, match = "Must provide `fasta`"):
                check_args(DummyNamespace(DUMMY_POSITIONS, [model], 
                                        evmutation_model = EVMUT_LOC))
                
            # Error if fasta cannot be found
            with pytest.raises(AssertionError, match = "Cannot find `fasta` file"):
                check_args(DummyNamespace(DUMMY_POSITIONS, [model], 
                                        evmutation_model = EVMUT_LOC,
                                        fasta = "./FAKE.txt"))
                
    # Error for incorrect models
    unknown_models = ["".join(random.choice(string.printable) for _ in range(20))
                    for _ in range(10)]
    for unknown_model in unknown_models:
        
        with pytest.raises(AssertionError, match = "Unrecognized models:"):
            check_args(DummyNamespace(DUMMY_POSITIONS, [unknown_model] + ALLOWED_MODELS,
                                    evmutation_model = EVMUT_LOC,
                                    fasta = FASTA_LOC,
                                    alignment = ALIGNMENT_LOC))
            
    # Error for no provided positions
    with pytest.raises(AssertionError, match = "Must specifiy at least one position"):
        check_args(DummyNamespace([], ALLOWED_MODELS,
                                evmutation_model = EVMUT_LOC,
                                fasta = FASTA_LOC,
                                alignment = ALIGNMENT_LOC))
        
    # Error for batch size too small
    with pytest.raises(AssertionError, match = "Batch size cannot be <1"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS,
                                evmutation_model = EVMUT_LOC,
                                fasta = FASTA_LOC,
                                alignment = ALIGNMENT_LOC,
                                batch_size = 0))
        
    # Error for no output directory
    with pytest.raises(AssertionError, match = "Directory does not exist:"):
        check_args(DummyNamespace(DUMMY_POSITIONS, ALLOWED_MODELS,
                                evmutation_model = EVMUT_LOC,
                                fasta = FASTA_LOC,
                                alignment = ALIGNMENT_LOC,
                                output = "./ADLKFJACLDKJAEC"))