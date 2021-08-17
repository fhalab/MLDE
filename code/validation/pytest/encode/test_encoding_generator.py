"""
This file contains functions that test the functionality of MLDE/code/support/encode/encoding_generator.py
"""
# Import necessary modules
import pytest
import time
import os
import shutil
import pickle
import subprocess
import warnings
import numpy as np
from itertools import product
from Bio import SeqIO

# Import the encoding generator
from ....encode.encoding_generator import EncodingGenerator
from ....encode.molbio_info import ALL_AAS
from ....encode.model_info import TAPE_MODEL_LOCATIONS, TRANSFORMER_INFO, N_LATENT_DIMS
from ....encode.georgiev_params import GEORGIEV_PARAMETERS

# Define the location of the test output folder
test_output_folder = "./code/validation/pytest/encode/test_output"

# Define allowed encodings
allowed_all = ("onehot", "georgiev", "resnet", "bepler", "unirep", "transformer",
               "lstm", "esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S",
               "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100", "esm1_t12_85M_UR50S",
               "esm1_t6_43M_UR50S", "prot_bert_bfd", "prot_bert")
allowed_learned = ("resnet", "bepler", "unirep", "transformer", "lstm")
allowed_learned_plusmsa = ("resnet", "bepler", "unirep", "transformer", "lstm",
                           "esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S",
                           "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100",
                           "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S",
                           "esm_msa1_t12_100M_UR50S", "prot_bert_bfd", "prot_bert")
allowed_learned_minusmsa = ("resnet", "bepler", "unirep", "transformer", "lstm",
                           "esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S",
                           "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100",
                           "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S",
                           "prot_bert_bfd", "prot_bert")
esm_models = ("esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S",
            "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100",
            "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S")             

# Write a function that compares encoder properties
def compare_encoder_properties(encoder1, encoder2):
    
    # Assert properties are the same where reasonable
    assert encoder1.n_positions_combined == encoder2.n_positions_combined
    assert encoder1.combi_space == encoder2.combi_space
    assert all(combo1 == combo2 for combo1, combo2 in
               zip(encoder1.all_combos, encoder2.all_combos))

# Define a function that tests _normalize_encodings()
def test_normalize_encodings():
    """
    This function simply tests where things passed in to _normalize_encodings
    will be correctly normalized.
    """    
    # Import function to test
    from ....encode.support_funcs import normalize_encodings
    
    # We expect an error if the incorrect shape is passed in
    with pytest.raises(AssertionError):
        normalize_encodings(np.random.rand(1000, 10))
        
    # Now define an input array and what we expect it to be coming out
    test_input = np.random.rand(1000, 7, 100)
    
    # Get the latter dimensions
    flat_test = np.empty([1000, 700])
    for i in range(1000):
        filler = []
        for row in test_input[i]:
            filler.extend(row)
        flat_test[i] = filler
        
    # Get means and standard deviations
    means = np.array([np.mean(flat_test[:, i]) for i in range(700)])
    stdevs = np.array([np.std(flat_test[:, i]) for i in range(700)])
    
    # Mean center and unit-scale
    center_scaled = np.array([(flat_test[:, i] - means[i])/stdevs[i]
                              for i in range(700)]).T
    
    # Reshape
    test_output = np.array([[center_scaled[i][j: j+100] for j in range(0, 700, 100)]
                            for i in range(1000)])
    
    # Assert all close
    assert np.allclose(test_output, normalize_encodings(test_input))
    
# This function checks the initialization stage of the EncodingGenerator
def test_EncodingGenerator_init():
    """
    This function will...
    1. Test to be sure we behave correctly under different circumstances: If learned
        encodings are used, then we should go through more checks (two other functions)
        than if georgiev or onehot are used. All should still achieve the same 
        final variables, however; an incorrect encoding should be reported as an error
    2. Make sure that learned encodings require target protein indices
    3. Make sure that onehot and georgiev require n_positions_combined
    4. Make sure an error is thrown if we have an unknown encoding
    """

    # Make sure errors are thrown if we don't have the correct variables for each encoding
    with pytest.raises(AssertionError, match = "Did not define n_positions_combined"):
        _ = EncodingGenerator("georgiev", "test",
                              output = test_output_folder)
    with pytest.raises(AssertionError, match = "Did not define n_positions_combined"):
        _ = EncodingGenerator("onehot", "test",
                              output = test_output_folder)
        
    # Loop over all embeddings and make sure we throw an error if we did not pass
    # in the correct arguments
    for encoding in allowed_learned_plusmsa:
        with pytest.raises(AssertionError, match = "Did not define target indices"):
            _ = EncodingGenerator(encoding, "test",
                                output = test_output_folder)
        
    # Make sure there is an error if we pass in an unknown encoding
    with pytest.raises(AssertionError, match = "Unknown encoding"):
        _ = EncodingGenerator("alkdjfhlaksdf", "test",
                              output = test_output_folder)
        
    # Define objects for the unlearned encodings
    georgiev_obj = EncodingGenerator("georgiev", "test", n_positions_combined = 3,
                                     output = test_output_folder)
    time.sleep(1)
    onehot_obj = EncodingGenerator("onehot", "test", n_positions_combined = 3,
                                     output = test_output_folder)
    compare_encoder_properties(georgiev_obj, onehot_obj)
    
    # Make sure the variables are the size we expect
    assert georgiev_obj.n_positions_combined == 3
    assert georgiev_obj.combi_space == 8000
    
    # Define objects for the learned encodings and make sure we arrive at the 
    # same variables as with georgiev and onehot encodings
    for encoding in allowed_learned_minusmsa:
        
        # Make the learned embedding object
        time.sleep(1)
        learned_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                        fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta",
                                        target_protein_indices = ("Q5", "L10", "G19"))
        
        # Make sure the embedding object has the same variables as the georgiev one
        compare_encoder_properties(learned_obj, georgiev_obj)
        
    # Do the same thing for the msa trasnfoer
    time.sleep(1)
    learned_obj = EncodingGenerator("esm_msa1_t12_100M_UR50S", "test", output = test_output_folder,
                                    fasta_path = "./code/validation/pytest/encode/test_data/DummyA2M.a2m",
                                    target_protein_indices = ("S6", "L10", "G19"))
        
    # Make sure the embedding object has the same variables as the georgiev one
    compare_encoder_properties(learned_obj, georgiev_obj)
        
    # Purge the test output folder
    shutil.rmtree(test_output_folder)

# Write a function that tests _process_input_fasta
def test_process_input_fasta():
    """
    This function only applies for learned embeddings from tape. The test will...
    1. Make sure all assertion errors in the process function are raised when
        appropriate
    2. Confirm that we get back the expected protein sequence
    """
    # Loop over all allowed embeddings
    for encoding in allowed_learned_plusmsa:
        
        # Make sure that we have an error if we can't find a file
        with pytest.raises(IOError, match = "Cannot locate './code/validation/pytest/encode/test_data/Dud.fasta'"):
            EncodingGenerator(encoding, "test", output = test_output_folder,
                              fasta_path = "./code/validation/pytest/encode/test_data/Dud.fasta",
                              target_protein_indices = ("Q5", "L10", "G19"))
        time.sleep(1)
        
        # Tests for non-msa transformer
        if encoding != "esm_msa1_t12_100M_UR50S":
            
            # Make sure we have an error if the fasta file has two sequences
            with pytest.raises(AssertionError, match = "Embedding generator can currently only handle 1 parent sequence"):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta2Seqs.fasta",
                                target_protein_indices = ("Q5", "L10", "G19"))
            time.sleep(1)
        
            # Make sure we have an error if we have forbidden characters
            with pytest.raises(AssertionError, match = "Forbidden character in input sequence."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                fasta_path = "./code/validation/pytest/encode/test_data/DummyFastaForbiddenCharacters.fasta",
                                target_protein_indices = ("Q5", "L10", "G19"))
            time.sleep(1)
        
            # Make sure the output sequence is correct
            embedding_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                            fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta",
                                            target_protein_indices = ("Q5", "L10", "G19"))
            assert embedding_obj.wt_seq == "VAFYQSTGHLMNDDDAAGGLIMNPPQDE"
        
        # Tests for msa transformer
        else:
            
            
            # Make sure we have an error if the fasta file has two sequences
            with pytest.raises(AssertionError, match = "Expected alignment, but received fasta"):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta",
                                target_protein_indices = ("S6", "L10", "G19"))
            time.sleep(1)
        
            # Make sure we have an error if we have forbidden characters
            with pytest.raises(AssertionError, match = "Forbidden character in input sequence."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                fasta_path = "./code/validation/pytest/encode/test_data/DummyA2MForbiddenCharacters.a2m",
                                target_protein_indices = ("S6", "L10", "G19"))
            time.sleep(1)
            
            # Make sure we have an error if we have forbidden characters
            with pytest.raises(AssertionError, match = "Forbidden character in input sequence."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                fasta_path = "./code/validation/pytest/encode/test_data/DummyA3MForbiddenCharacters.a3m",
                                target_protein_indices = ("S6", "L10", "G19"))
            time.sleep(1)
        
            # Make sure the output sequence is correct
            embedding_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                            fasta_path = "./code/validation/pytest/encode/test_data/DummyA2M.a2m",
                                            target_protein_indices = ("S6", "L10", "G19"))
            assert embedding_obj.wt_seq[0][1] == "YSTGHLM-NDDDAGGLIMNPQD"
            time.sleep(1)
            
            embedding_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                            fasta_path = "./code/validation/pytest/encode/test_data/DummyA3M.a3m",
                                            target_protein_indices = ("S6", "L10", "G19"))
            assert embedding_obj.wt_seq[0][1] == "YSTGHLM-NDDDAGGLIMNPQD"
        
        # Purge the test output folder
        shutil.rmtree(test_output_folder)

# Write a function that tests _check_indices()
def test_check_indices():
    """
    This function checks the below:
    1. The index_splitter correctly splits amino acid positions and letters
    2. The index_splitter throws an error if the input format is wrong
    3. The correct python indices are generated from the input protein indices
    4. If amino acid names don't match up, we throw an error
    5. The correct wild type amino acids are pulled
    6. Throw an error if indices are not pased in in order
    7. Throw an error if duplicate indices are passed in
    """
    # Define the target indices and expected outputs
    expected_aas = ("S", "L", "G")
    expected_python_inds = (5, 9, 18)
    expected_protein_inds = ("S6", "L10", "G19")
    
    # Make some bad inputs
    bad_in1 = ("6S", "10L", "19G")
    bad_in2 = ("S6", "10L", "19G")
    bad_in3 = ("S_6", "L_10", "G_19")
    
    bad_in4 = ("A6", "L10", "G19")
    bad_in5 = ("S6", "D10", "G19")
    bad_in6 = ("S6", "L10", "Z19")
    bad_in7 = ("A6", "D10", "E19")
    
    bad_in8 = ("L10", "S6", "G19")
    bad_in9 = ("G19", "L10", "S6")
    
    bad_in10 = ("S6", "S6", "G19")
    bad_in11 = ("S6", "S6", "S6")
    
    bad_in12 = ("E0", "L10", "G19")
    bad_in13 = ("S6", "L10", "G199")
        
    # Subfunction to handle tests
    def check_input_inds(encoding, fasta_path):
        
        # Build an Embedding object
        embedding_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                        fasta_path = fasta_path,
                                        target_protein_indices = expected_protein_inds)
        time.sleep(1)
        
        # Make sure we correctly split amino acids
        assert all((aa == expected_aas[i] and ind == expected_python_inds[i] and pind == expected_protein_inds[i])
                   for i, (aa, ind, pind) in enumerate(zip(embedding_obj.wt_aas, embedding_obj.target_python_inds,
                                                           embedding_obj.target_protein_indices)))
        
        # Make sure we throw an error if the input format is wrong
        for bad_in in (bad_in1, bad_in2, bad_in3):
            with pytest.raises(AssertionError, match = "Unrecognizable protein index."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                  fasta_path = fasta_path,
                                  target_protein_indices = bad_in)
        
        # Make sure an error is thrown if the amino acid identities don't match
        # with the expected
        for bad_in in (bad_in4, bad_in5, bad_in6, bad_in7):
            with pytest.raises(AssertionError, match = "Requested positions not found."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                  fasta_path = fasta_path,
                                  target_protein_indices = bad_in)
        
        # Now make sure we throw an error if amino acid indices aren't passed in
        # in order
        for bad_in in (bad_in8, bad_in9):
            with pytest.raises(AssertionError, match = "Out of order indices."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                  fasta_path = fasta_path,
                                  target_protein_indices = bad_in)
        
        # Now make sure we throw an error if there are duplicates indices found
        for bad_in in (bad_in10, bad_in11):
            with pytest.raises(AssertionError, match = "Duplicate indices identified."):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                  fasta_path = fasta_path,
                                  target_protein_indices = bad_in)
                
        for bad_in in (bad_in12, bad_in13):
            with pytest.raises(AssertionError, match = "Out of range AA index"):
                EncodingGenerator(encoding, "test", output = test_output_folder,
                                  fasta_path = fasta_path,
                                  target_protein_indices = bad_in)
    
    # Loop over all embeddings
    for encoding in allowed_learned_plusmsa:    
        
        # Define the path to the fasta
        fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta"
        if encoding != "esm_msa1_t12_100M_UR50S":
            
            # Run checks for non-msa transformer
            check_input_inds(encoding, fasta_path)
            
        else:
            
            # Check both A2M and A3M inputs for the MSA transformer
            fasta_path1 = "./code/validation/pytest/encode/test_data/DummyA2M.a2m"
            check_input_inds(encoding, fasta_path1)
            fasta_path2 = "./code/validation/pytest/encode/test_data/DummyA3M.a3m"
            check_input_inds(encoding, fasta_path2)
            
            # Run specific check for the MSA transformer
            with pytest.raises(AssertionError, match = "Expected alignment, but received fasta"):
                embedding_obj = EncodingGenerator(encoding, "test", output = test_output_folder,
                                                fasta_path = fasta_path,
                                                target_protein_indices = expected_protein_inds)
        
                
    # Purge the test output folder
    shutil.rmtree(test_output_folder)

# Write a function that tests build_combo_dicts
def test_build_combo_dicts():
    """
    This function makes sure that the combo_to_index and index_to_combo dictionaries
    generated and saved during initialization are accurate
    """
    # Make sure all_aas is the right length
    assert len(set(ALL_AAS)) == 20
    
    # Design a set of all possible combinations for 2, 3 and 4 site libraries
    lib_sizes = (2, 3, 4)
    expected_lengths = (400, 8000, 160000)
    
    # Loop over possible library sizes
    for lib_ind, lib_size in enumerate(lib_sizes):
        
        # Generate a test library
        test_lib = list(product(ALL_AAS, repeat=lib_size))
        
        # Loop over all encodings
        for encoding in allowed_all:
            
            # Define the input files
            if encoding != "esm_msa1_t12_100M_UR50S":
                fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta"
            else:
                fasta_path = "./code/validation/pytest/encode/test_data/DummyA2M.a2m"
                
            # Make the encoder
            encoder = EncodingGenerator(encoding, "test", output = test_output_folder,
                                        fasta_path = fasta_path,
                                        target_protein_indices = ("A2", "S6", "L10", "G19")[:lib_size],
                                        n_positions_combined = lib_size)
            time.sleep(1)
        
            # Load the index to combo and combo to index dictionaries
            with open(os.path.join(encoder.encoding_output, f"test_{encoding}_ComboToIndex.pkl"), "rb") as f:
                combo_to_ind = pickle.load(f)
            with open(os.path.join(encoder.encoding_output, f"test_{encoding}_IndexToCombo.pkl"), "rb") as f:
                ind_to_combo = pickle.load(f)
                
            # Make sure the dictionaries are in agreement
            assert all(ind_to_combo[val] == key for key, val in combo_to_ind.items())
            
            # Make sure the lengths of everything is as expected
            assert len(test_lib) == expected_lengths[lib_ind]
            assert len(test_lib) == len(ind_to_combo)
            assert len(test_lib) == len(combo_to_ind)
            assert len(test_lib) == len(encoder.all_combos)
            
            # Make sure our test library and the all_combos libraries align
            assert all(ind_to_combo[i] == "".join(combo) for i, combo in enumerate(test_lib))
            assert all(ind_to_combo[i] == "".join(combo) for i, combo in enumerate(encoder.all_combos))
            
            # Purge the test output folder
            shutil.rmtree(test_output_folder)

# Write a function to be sure we are correctly making fastas
def test_make_fastas():
    """
    This function tests...
    1. To be sure that the correct mutations are made in the correct locations
        of the wild type sequence for fasta files of different library sizes
    2. To be sure that batching is being performed appropriately and we are not
        either cutting out sequences or adding sequences
    3. To be sure that the combo_to_index and index_to_combo dictionaries
        generated and saved during initialization are accurate
    """
    # Design a set of all possible combinations for 2, 3 and 4 site libraries
    lib_sizes = (2, 3, 4)
    expected_lengths = (400, 8000, 160000)
    
    # Defie a number of different batch sizes (making them awkard on purpose)
    batch_sizes = (1, 3, 4, 5, 7, 10, 13, 59)
    
    # Define the positions to mutate
    mutated_positions = ("A2", "S6", "L10", "G19")
    expected_mut_inds = (1, 5, 9, 18)
    
    # Loop over possible library sizes
    for lib_ind, lib_size in enumerate(lib_sizes):
                
        # Generate a test library
        test_lib = list(product(ALL_AAS, repeat=lib_size))
                
        # Loop over all encodings
        for encoding in allowed_all:
            
            # Define the input files
            if encoding != "esm_msa1_t12_100M_UR50S":
                fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta"
            else:
                fasta_path = "./code/validation/pytest/encode/test_data/DummyA2M.a2m"
            
            # Loop over batch sizes
            for batch_size in batch_sizes:
            
                # Make the encoder
                encoder = EncodingGenerator(encoding, "test", output = test_output_folder,
                                            fasta_path = fasta_path,
                                            target_protein_indices = mutated_positions[:lib_size],
                                            n_positions_combined=lib_size)
                time.sleep(1)
                
                # Load the index to combo and combo to index dictionaries
                with open(os.path.join(encoder.encoding_output, f"test_{encoding}_ComboToIndex.pkl"), "rb") as f:
                    combo_to_ind = pickle.load(f)
                with open(os.path.join(encoder.encoding_output, f"test_{encoding}_IndexToCombo.pkl"), "rb") as f:
                    ind_to_combo = pickle.load(f)
                    
                # Make sure the dictionaries are in agreement
                assert all(ind_to_combo[val] == key for key, val in combo_to_ind.items())
                
                # Make sure the lengths of everything is as expected
                assert len(set(test_lib)) == expected_lengths[lib_ind]
                assert len(test_lib) == len(ind_to_combo)
                assert len(test_lib) == len(combo_to_ind)
                assert set(test_lib) == set(encoder.all_combos)
                
                # Make sure our test library and the all_combos libraries align
                assert all(ind_to_combo[i] == "".join(combo) for i, combo in enumerate(test_lib))
                assert all(ind_to_combo[i] == "".join(combo) for i, combo in enumerate(encoder.all_combos))
                
                # If this is not a tape encoding we are done here
                if encoding not in allowed_learned:
                    continue
                
                # Make fastas
                fasta_files = encoder._build_fastas(batch_size)
                
                # Now load all fasta files
                mutated_seqs = []
                for fasta_file in fasta_files:
                    
                    # Load the fasta file
                    with open(fasta_file, "r") as f:
                        
                        # Open the fasta file and extract all sequences
                        fasta_seqs = list(SeqIO.parse(f, "fasta"))
            
                    # Convert the full sequence to uppercase and record
                    mutated_seqs.extend([str(seq.seq) for seq in fasta_seqs])
                    
                # Make sure the length is as expected
                assert len(mutated_seqs) == expected_lengths[lib_ind]
                    
                # Loop over the sequences and make sure the amino acids match
                # the combinations
                for seqid, seq in enumerate(mutated_seqs):
                    
                    # Pull the amino acids at the correct sequence positions
                    mutated_aas = [seq[ind] for ind in expected_mut_inds[:lib_size]]
                    
                    # Make sure the mutated_aas match the combination expected
                    assert all(mutant_aa == expected_aa for mutant_aa, expected_aa
                               in zip(encoder.all_combos[seqid], mutated_aas))
                    
    # Purge the test output folder
    shutil.rmtree(test_output_folder)
                
# Test the generate_encodings function
def test_generate_encodings():
    """
    This is a beefy test function. Using it, we test the functions:
    1. _generate_onehot()
    2. _generate_georgiev()
    3. _generate_tape()
    4. _generate_transformer()
    
    This function is tested last because it relies on all of the preceding functions
    working. For each of the above encoding functions, we test whether the encodings 
    we get from a subset of all encodings align with what we would have arrived
    at performing encoding manually (or as close to it as we can for the learned
    embeddings). This relies pulling the expected index of the encoding using 
    the appropriate dictionaries. We also test to be sure there are no duplicate
    encodings in the returned arrays. All of these tests are performed over multiple
    batch sizes, but only for libraries of size 2 and 3 (For the sake of compute time)  
    """
    # Try to import torch. Set flags to see if we are going to run the esm models
    # or just the tape models
    try:
        import torch
        models_to_test = esm_models
    except ModuleNotFoundError:
        models_to_test = allowed_learned
        warnings.warn("Could not load torch. This is expected if running in "
                      "the mlde environment. It is not expected if running in "
                      "mlde2.")
    
    # Design a set of all possible combinations for 2 and 3 site libraries
    lib_sizes = (2, 3)
    expected_lengths = (400, 8000)
    
    # Defie a number of different batch sizes (making them awkard on purpose)
    batch_sizes = (1, 3, 4, 13, 59)
    
    # Define the expected shapes
    expected_shapes = {"onehot": 20,
                       "georgiev": 19}
    expected_shapes.update(N_LATENT_DIMS)
    
    # Define the positions to mutate
    mutated_positions = ("A2", "Q5", "G19")
    expected_mut_inds = (1, 4, 18)
    
    # Define the variants that we will test
    tested_vars = ("ADQ", "GHI", "WYD", "IFK", "PRS")
    
    # Define the expected onehot encodings
    expected_onehot = np.array([
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            ],
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        ],
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            ],
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]
        ])
    
    # Define the expected Georgiev arrays
    expected_georgiev = np.empty([10, 3, 19])
    for i, combo in enumerate(tested_vars):
        for j, char in enumerate(combo):
            for k, param in enumerate(GEORGIEV_PARAMETERS):
                expected_georgiev[i, j, k] = param[char]
                expected_georgiev[i + 5, j, k] = param[char]
    
    # Get the location of the fasta containing the test variants
    test_fasta = "./code/validation/pytest/encode/test_data/DummyFastaEmbeddingTest.fasta"
    seqlen = len(str(next(SeqIO.parse(test_fasta, "fasta")).seq))
    
    # Define a dictionary for storing expected embedding results
    expected_results = {"onehot": expected_onehot,
                        "georgiev": expected_georgiev}
    
    # Define an environment for the esm submodule. It can have issues if numpy
    # isn't imported before torch OR if the below environment variable isn't set
    my_env = os.environ.copy()
    my_env['MKL_THREADING_LAYER'] = 'GNU'
    
    # Run tape for all models using the test variants
    for encoding in models_to_test:
        
        # Get the save location
        savename = "DummyFastaEmbeddingTest.pkl"
        save_loc = "./code/validation/pytest/encode/test_data/EncodingOutput/"
        
        # If the encoding is a tape model, generate expected data down this route
        if encoding in allowed_learned:
        
            # Run tape
            tape_savename = f"{save_loc}{savename}"
            _ = subprocess.run(["tape-embed", test_fasta,
                                encoding, "--load-from", TAPE_MODEL_LOCATIONS[encoding],
                                "--output", tape_savename], check = True)
            
            # Load results and store
            with open(tape_savename, "rb") as f:
                tape_results = pickle.load(f)
                
            # Restructure tape results
            embedding_results = [result[0] for result in tape_results]
            
        # If the encoding is esm, generate expected data down this route
        elif "esm" in encoding:
            
            # Run ESM
            _ = subprocess.run(["python", "./code/esm/extract.py", encoding,
                                test_fasta, save_loc, "--include", "per_tok",
                                "--toks_per_batch", "29"], check = True, env = my_env)
            
            # Open results and store
            target_layer = TRANSFORMER_INFO[encoding][1]
            esm_results = [torch.load(f"{save_loc}TestSeq{i + 1}.pt") for i in range(10)]
            embedding_results = [result["representations"][target_layer].numpy()
                                 for result in esm_results]
                   
                   
        # Any other encoding this is an error
        else:
            assert False, "Unknown encoding tested"   
                     
        # Store results
        expected_learned = np.empty([len(tested_vars) * 2, 3, expected_shapes[encoding]])
        for i, result in enumerate(embedding_results):
            for j, expected_mut_ind in enumerate(expected_mut_inds):
                assert result.shape == (seqlen, expected_shapes[encoding]) # (L_seq, emb_dim)
                expected_learned[i, j] = result[expected_mut_ind]
        expected_results[encoding] = expected_learned
    
    # Loop over possible library sizes
    for lib_ind, lib_size in enumerate(lib_sizes):
                
        # Generate a test library
        test_lib = list(product(ALL_AAS, repeat=lib_size))
        
        # Get the back range of expected results
        if lib_size == 2:
            expected_results_range = np.arange(0, 5)
        else:
            expected_results_range = np.arange(5, 10)
                        
        # Loop over all encodings
        for encoding in models_to_test:
            
            # Loop over batch sizes
            for batch_size in batch_sizes:
            
                # Make the encoder
                encoder = EncodingGenerator(encoding, "test", output = test_output_folder,
                                            fasta_path = "./code/validation/pytest/encode/test_data/DummyFasta.fasta",
                                            target_protein_indices = mutated_positions[:lib_size],
                                            n_positions_combined=lib_size)
                encoder.generate_encodings(n_batches=batch_size, batch_size=1)
                time.sleep(1)
                
                # Load the index to combo and combo to index dictionaries
                with open(os.path.join(encoder.encoding_output, f"test_{encoding}_ComboToIndex.pkl"), "rb") as f:
                    combo_to_ind = pickle.load(f)
                with open(os.path.join(encoder.encoding_output, f"test_{encoding}_IndexToCombo.pkl"), "rb") as f:
                    ind_to_combo = pickle.load(f)
                    
                # Make sure that the lengths fo combo to ind and ind to combo are accurate
                assert len(combo_to_ind) == expected_lengths[lib_ind]
                assert len(ind_to_combo) == expected_lengths[lib_ind]
                
                # Make sure the test library is the same as the actual library
                assert [test_combo == actual_combo for test_combo, actual_combo
                        in zip(test_lib, encoder.all_combos)]
                assert len(test_lib) == len(encoder.all_combos)
                assert len(set(encoder.all_combos)) == len(encoder.all_combos)
                assert len(set(encoder.all_combos)) == expected_lengths[lib_ind]
                
                # Make sure that combo_to_ind and ind_to_combo are equivalent
                assert all(combo_to_ind[val] == key for key, val in ind_to_combo.items())
                    
                # Load the unnormalized encodings 
                unnorm_encodings = np.load(os.path.join(encoder.encoding_output, 
                                                        f"{encoder.protein_name}_{encoder.encoding}_UnNormalized.npy"))
      
                # Make sure that the shape of the encodings is what we expect
                assert (unnorm_encodings.shape == (expected_lengths[lib_ind], 
                                                  lib_size,
                                                  expected_shapes[encoding]))
                
                # Find the indices for test combinations
                test_comb_inds = np.array([combo_to_ind[combo[:lib_size]]
                                           for combo in tested_vars])
                                
                # Make sure that the expected and actual results are equivalent
                assert np.array_equal(unnorm_encodings[test_comb_inds], 
                                      expected_results[encoding][expected_results_range, :lib_size])
                
                # Get the flat unnormalized encodings
                flat_unnorm = np.reshape(unnorm_encodings, 
                                         [len(unnorm_encodings), 
                                          np.product(unnorm_encodings.shape[1:])])
                
                # Make sure every row is unique
                assert len(flat_unnorm) == len(np.unique(flat_unnorm, axis = 0))
                
                # If this is a onehot encoding, make sure all rows add up to the 
                # number of positions in library. Make sure all columns add up to
                # the number of combinations / 20. Afterward, continue
                if encoding == "onehot":
                   
                    # Add across all rows and columns
                    row_sum = flat_unnorm.sum(axis = 1)
                    col_sum = flat_unnorm.sum(axis = 0)
                    
                    # Make sure the shapes of rows and columns are correct
                    assert len(row_sum.shape) == 1
                    assert len(col_sum.shape) == 1
                    assert len(row_sum) == expected_lengths[lib_ind]
                    assert len(col_sum) == lib_size * 20
                    
                    # Make sure the colums and rows have the appropriate additions
                    assert np.all(row_sum == lib_size)
                    assert np.all(col_sum == expected_lengths[lib_ind] / 20)
                    
                    continue
                
                # Load the normalized encodings
                norm_encodings = np.load(os.path.join(encoder.encoding_output, 
                                                      f"{encoder.protein_name}_{encoder.encoding}_Normalized.npy"))
                
                # Make sure the shape is what we expect
                assert norm_encodings.shape == unnorm_encodings.shape
                
                # Make sure the array is different from the unnormalized encodings
                assert not np.array_equal(norm_encodings, unnorm_encodings)
                
                # Now make sure that we are unit-scaled, mean-centered
                flat_norm = np.reshape(norm_encodings, 
                                       [len(norm_encodings), 
                                        np.product(norm_encodings.shape[1:])])
                test_mean = flat_norm.mean(axis=0)
                test_stdev = flat_norm.std(axis = 0)
                
                # Make sure all rows are unique
                assert len(flat_norm) == len(np.unique(flat_norm, axis = 0))
                
                # Make sure our mean and standard deviation vectors are the 
                # expected length
                assert len(test_mean) == expected_shapes[encoding] * lib_size
                assert len(test_stdev) == expected_shapes[encoding] * lib_size
                
                # Make sure the mean is roughly 0 and the standard deviation is 
                # roughly 1
                assert np.all(np.abs(test_mean) < 1e-2)
                assert np.all(np.abs(np.ones(len(test_stdev)) - test_stdev) < 1e-4)
                
                # Purge the test output folder
                shutil.rmtree(test_output_folder)