# Import pytest and modules for testing 
import pytest
from ....encode.transformer_classes import TRANSFORMER_TO_CLASS

# Import other needed objects/modules
from ....encode.molbio_info import ALL_AAS
import torch
import string
import itertools
import random
import numpy as np

# Define globals
GENERIC_N_TESTS = 10

# Define a maximum depth for alignments
ALIGNMENT_DEPTH = 10

# Make a set of combo sizes to try
COMBO_SIZES = (1, 2, 3, 4)

def generate_test_objects(include_mask = True):
    
    # Loop over all models
    for model_name, model_class in TRANSFORMER_TO_CLASS.items():
            
        # Build the object
        transformer = model_class(model_name)
        
        # Define the allowed characters and make combo lists
        if include_mask:
            allowed_chars = tuple(list(ALL_AAS) + [transformer.mask_string])
        else:
            allowed_chars = ALL_AAS
        all_combo_lists = [list(itertools.product(allowed_chars, repeat = size))
                           for size in COMBO_SIZES]
    
        # Loop over all combo sizes
        for i, combo_size in enumerate(COMBO_SIZES):
            
            # Get the appropriate combo list
            combo_list = all_combo_lists[i]        
            
            # Get incorrect combo sizes
            incorrect_combo_sizes = [size for size in COMBO_SIZES if size != combo_size]
            
            # Loop over a set number of tests
            for _ in range(GENERIC_N_TESTS):
                
                # Pull a combo at random to use as the parent and assign
                # random positions. 
                parent_combo = random.choice(combo_list)
                target_positions = np.random.randint(1, 1000, size = combo_size)
                
                # Build a set of bad combos and targets
                incorrect_size = random.choice(incorrect_combo_sizes)
                bad_parent_combo = "".join([random.choice(string.ascii_uppercase) for _ in range(incorrect_size)])
                bad_targets = np.random.randint(1, 1000, size = incorrect_size)
                
                # Get a list of incorrect combos
                incorrect_combo_list = all_combo_lists[incorrect_size - 1]
                
                # Yield all information as a dictionary
                yield {"Model": transformer,
                       "ComboList": combo_list,
                       "ParentCombo": parent_combo,
                       "TargetPositions": target_positions,
                       "BadParentCombo": bad_parent_combo,
                       "BadTargets": bad_targets,
                       "IncorrectComboList": incorrect_combo_list,
                      "ModelName": model_name}

def build_seq_data(test_data):

    # Build a random sequence
    seq_len = np.random.randint(10, 1022)
    random_sequence = "".join(random.choice(ALL_AAS) for _ in range(seq_len))

    # Choose target and parent positions
    target_positions = np.random.choice(np.arange(0, seq_len), 
                                        size = len(test_data["ParentCombo"]),
                                        replace = False)
    parent_combo = [random_sequence[pos] for pos in target_positions]
    target_positions += 1

    # If a transformer, build the rest of the alignment
    if test_data["ModelName"] == "esm_msa1_t12_100M_UR50S":
        alignment_depth = np.random.choice(ALIGNMENT_DEPTH)
        alignment_seqs = [[str(i), "".join(random.choice(ALL_AAS) for _ in range(seq_len))]
                        for i in range(alignment_depth)]
        random_sequence = [["Ref", random_sequence]] + alignment_seqs

    # Return the random sequence
    return random_sequence, parent_combo, target_positions

def test_array_split():
    """
    Makes sure that the numpy array split function is working and
    keeping everything in order. 
    """
    # Define a set of batch sizes to test
    batch_sizes = np.random.randint(1, 100, GENERIC_N_TESTS)

    # Build combos
    all_combos = list(list(combo) for combo in itertools.product(ALL_AAS, repeat = 4))

    # Loop over tests
    for n_batches in batch_sizes:

        # Split combos
        split_combos = np.array_split(all_combos, n_batches)

        # Join combos
        rejoined_combos = np.concatenate(split_combos).tolist()

        # Make sure combos match the input
        assert all_combos == rejoined_combos

# Makes sure the check_combo_info function behaves as expected
def test_abstract_check_combo_info():
    
    # Loop over all tests
    for test_data in generate_test_objects():

        # Define the model
        transformer = test_data["Model"]

        # Make sure good combos work. 
        transformer._check_combo_info(test_data["TargetPositions"],
                                      test_data["ComboList"],
                                      test_data["ParentCombo"])

        # Make sure we fail if the wrong parent combo size is passed in
        with pytest.raises(AssertionError, match = "Mismatch between combo length and N targets"):
            transformer._check_combo_info(test_data["TargetPositions"],
                                          test_data["ComboList"],
                                          test_data["BadParentCombo"])

        # Make sure we fail if the wrong positions are passed in
        with pytest.raises(AssertionError, match = "Mismatch between combo length and N targets"):
            transformer._check_combo_info(test_data["BadTargets"],
                                          test_data["ComboList"],
                                          test_data["ParentCombo"])

        # Make sure we fail if the wrong combo list is passed in
        with pytest.raises(AssertionError, match = "Error in combo creation"):
            transformer._check_combo_info(test_data["TargetPositions"],
                                          test_data["IncorrectComboList"],
                                          test_data["ParentCombo"])

# ESM and Protbert (non-msa)
def test_esmprot_check_base_tokenization():
    
    # Loop over tests
    for test_data in generate_test_objects():
        
        # Skip if this is the MSA transformer
        if test_data["ModelName"] == "esm_msa1_t12_100M_UR50S":
            continue
            
        # Get a sequence, combo, and target positions for running tests
        random_sequence, parent_combo, target_positions = build_seq_data(test_data)
                
        # Make a base tokenization
        model = test_data["Model"]
        base_tokenization = model._build_base_tokenization(random_sequence)
        
        # Error if the tokenization dimensionality is off
        with pytest.raises(AssertionError, match = "Expected 1 element in first dimension"):
            model._check_base_tokenization(torch.rand(2, 1), random_sequence, 
                                        target_positions, parent_combo)
        
        with pytest.raises(AssertionError, match = "Incorrect token dim"):
            model._check_base_tokenization(torch.rand(1, 1, 1), random_sequence, 
                                        target_positions, parent_combo)
            
        # Error if the combo and mutant positions are off
        fake_parent = [None] * len(parent_combo)
        for i, parent_aa in enumerate(parent_combo):
            allowed_muts = [mut for mut in ALL_AAS if mut != parent_combo[i]]
            fake_parent[i] = random.choice(allowed_muts)
            
        with pytest.raises(AssertionError, match = "Unaligned parent combo and mutant positions"):
            model._check_base_tokenization(base_tokenization, random_sequence,
                                        target_positions, fake_parent)
            
        # Error if the sequence length is off
        bad_tokenization = torch.cat((base_tokenization, base_tokenization), axis = 1)
        with pytest.raises(AssertionError, match = "Expect addition of cls token"):
            model._check_base_tokenization(bad_tokenization, random_sequence,
                                        target_positions, parent_combo)
        
        # Error if cls isn't first
        no_cls = torch.clone(base_tokenization)
        no_cls[:, 0] = model.tok_to_idx[model.eos_string]
        with pytest.raises(AssertionError, match = "Expect addition of cls"):
            model._check_base_tokenization(no_cls, random_sequence,
                                           target_positions, parent_combo)
            
        # Pass in a bad sequence. We should error.
        seqlen = len(random_sequence)
        bad_seq = [random.choice(ALL_AAS) for _ in range(seqlen)]
        with pytest.raises(AssertionError, match = "Tokenization does not represent sequence"):
            model._check_base_tokenization(base_tokenization, bad_seq,
                                        target_positions, parent_combo)
            
def test_msa_check_base_tokenization():
    
    # Loop over tests
    for test_data in generate_test_objects():
        
        # Skip if this is not the MSA transformer
        if test_data["ModelName"] != "esm_msa1_t12_100M_UR50S":
            continue
            
        # Get a sequence, combo, and target positions for running tests
        random_sequence, parent_combo, target_positions = build_seq_data(test_data)
                
        # Make a base tokenization
        model = test_data["Model"]
        base_tokenization = model._build_base_tokenization(random_sequence)
        
        # Error if the tokenization dimensionality is off
        with pytest.raises(AssertionError, match = "Expected 1 element in first dimension"):
            model._check_base_tokenization(torch.rand(2, 1, 1), random_sequence, 
                                        target_positions, parent_combo)
            
        with pytest.raises(AssertionError, match = "Incorrect token dim"):
            model._check_base_tokenization(torch.rand(1, 1), random_sequence, 
                                        target_positions, parent_combo)
            
        # Error if the combo and mutant positions are off
        fake_parent = [None] * len(parent_combo)
        for i, parent_aa in enumerate(parent_combo):
            allowed_muts = [mut for mut in ALL_AAS if mut != parent_combo[i]]
            fake_parent[i] = random.choice(allowed_muts)
            
        with pytest.raises(AssertionError, match = "Unaligned parent combo and mutant positions"):
            model._check_base_tokenization(base_tokenization, random_sequence,
                                        target_positions, fake_parent)
            
        # Error if the number of alignments is off
        bad_tokenization = torch.cat((base_tokenization, base_tokenization), axis = 1)
        with pytest.raises(AssertionError, match = "Incorrect tokenization of alignments"):
            model._check_base_tokenization(bad_tokenization, random_sequence,
                                        target_positions, parent_combo)
        
        # Error if the sequence length is off
        bad_tokenization2 = torch.cat((base_tokenization, 
                                    torch.ones(*base_tokenization.shape, dtype = torch.long)),
                                    axis = 2)
        with pytest.raises(AssertionError, match = "Expect addition of cls. Refseq length off."):
            model._check_base_tokenization(bad_tokenization2, random_sequence,
                                        target_positions, parent_combo)
        
        # Error if cls isn't first
        no_cls = torch.clone(base_tokenization)
        no_cls[:, :, 0] = model.tok_to_idx[model.eos_string]
        with pytest.raises(AssertionError, match = "Expect addition of cls"):
            model._check_base_tokenization(no_cls, random_sequence,
                                        target_positions, parent_combo)
            
        # Pass in a bad sequence. We should error.
        seqlen = len(random_sequence[0][1])
        bad_seq = [random.choice(ALL_AAS) for _ in range(seqlen)]
        random_sequence[0][1] = bad_seq
        with pytest.raises(AssertionError, match = "Tokenization does not represent alignment"):
            model._check_base_tokenization(base_tokenization, random_sequence,
                                        target_positions, parent_combo)
        

# This is the only "computational" function we can really test here, as it is
# the only one with predetermined outcome. All of the other computations rely on
# this function being accurate, however, 
def test_abstract_build_mutant_tokens():
    
    # Loop over all tests
    for test_data in generate_test_objects():
        
        # Run with full_col on and off
        for full_col in (False, True):
            
            # Build sequence data
            random_sequence, parent_combo, target_positions = build_seq_data(test_data)

            # Build the mutant tokens for the test.
            all_tokens = test_data["Model"]._build_mutant_tokens(random_sequence,
                                                                target_positions,
                                                                test_data["ComboList"],
                                                                parent_combo,
                                                                full_col = full_col).cpu().numpy()

            # Assign the mutant tokens and sequence of the reference sequence
            if test_data["ModelName"] == "esm_msa1_t12_100M_UR50S":
                refseq = random_sequence[0][1]
                mutant_tokens = all_tokens[:, 0]
            else:
                mutant_tokens = all_tokens
                refseq = random_sequence


            # Extract the mutated tokens and translate to combos
            idx_to_tok = {val: key for key, val in test_data["Model"].tok_to_idx.items()}
            extracted_mutant_tokens = mutant_tokens[:, target_positions]
            extracted_mutant_combos = [tuple(idx_to_tok[idx] for idx in mut_tok_array)
                                    for mut_tok_array in extracted_mutant_tokens]
            assert extracted_mutant_combos == test_data["ComboList"]

            # Everything after the first token should match the sequence.
            mutant_seqs = [[idx_to_tok[idx] for idx in mut_tok_array]
                        for mut_tok_array in mutant_tokens]

            # Every character should match the original sequence except for the mutated
            # ones
            mutant_ind_set = set(target_positions)
            for mutant_seq in mutant_seqs:
                assert all(mutant_seq[i] == og_char for i, og_char in enumerate(refseq, 1)
                        if i not in mutant_ind_set)

            # Error if we pass in repeat combos. Make as many repeats as there are correct combos.
            repeat_combos = [test_data["ParentCombo"] for _ in range(len(test_data["ComboList"]))]
            with pytest.raises(AssertionError, match = "Non-unique combos found"):
                test_data["Model"]._build_mutant_tokens(random_sequence,
                                                    target_positions,
                                                    repeat_combos,
                                                    parent_combo,
                                                    full_col = full_col)
                
            # Additional checks for the msa transformer
            if test_data["ModelName"] == "esm_msa1_t12_100M_UR50S":
                
                # Loop over all combos
                for combo_ind, combo in enumerate(test_data["ComboList"]):
                    
                    # Find positions that have a mask token
                    masked_positions = [target_positions[char_ind] for char_ind, char in enumerate(combo)
                                    if char == test_data["Model"].mask_string]
                    
                    # Continue if no masked positions
                    if len(masked_positions) == 0:
                        continue
                    
                    # If full column is on, then mask tokens should be applied over a full column
                    mask_token = test_data["Model"].tok_to_idx[test_data["Model"].mask_string]
                    if full_col:
                        assert np.all(all_tokens[combo_ind, :, masked_positions].flatten() == mask_token)
                        
                    # If full column is off, then only the reference sequence should be masked
                    else:
                        assert np.all(all_tokens[combo_ind, 0, masked_positions].flatten() == mask_token)
                        assert np.all(all_tokens[combo_ind, 1:, masked_positions].flatten() != mask_token)