# Import pytest and the functions to be tested
import pytest
from ....encode.support_funcs import (check_args, process_alignment, build_old_to_new, 
                                      load_alignment, translate_target_positions)
from ....encode.model_info import N_LATENT_DIMS

# Load pytest and the Sequence loader
import pytest
import os
import string
import random
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

###############################################################################
################################# GLOBALS #####################################

# Define all encodings, just learned, and just regular
REGULAR_ENCODINGS = ("onehot", "georgiev")
LEARNED_ENCODINGS = tuple(N_LATENT_DIMS.keys())
ALL_ENCODINGS = tuple(list(LEARNED_ENCODINGS) + list(REGULAR_ENCODINGS))

# Define characters that can appear in a processed alignment
PROCESSED_CHAR_OPTS_LIST = list(string.ascii_uppercase) + ["-"]
PROCESSED_CHAR_OPTS = set(PROCESSED_CHAR_OPTS_LIST)

# Define characters that appear in an unprocessed alignment
UNPROCESSED_CHAR_OPTS_LIST = list(string.ascii_lowercase) + [".", "*"]
UNPROCESSED_CHAR_OPTS = set(UNPROCESSED_CHAR_OPTS_LIST)

# Define deletekeys
DELETEKEYS = {char: None for char in UNPROCESSED_CHAR_OPTS}

# Set the number of alignment tests
N_ALIGNMENT_TESTS = 100000

# Create a set of variables for building random alignments
REFSEQ_LENGTHS = np.random.randint(24, 1024, N_ALIGNMENT_TESTS)
LOWERCASE_PROBS = np.random.rand(N_ALIGNMENT_TESTS)
DASH_PROBS = np.random.rand(N_ALIGNMENT_TESTS)
DOT_PROBS = np.random.rand(N_ALIGNMENT_TESTS) / 2
ALIGNMENT_SIZES = np.random.randint(1, 20, N_ALIGNMENT_TESTS)
N_TARGET_POSITIONS = np.random.randint(1, 8, N_ALIGNMENT_TESTS)

# Define the output location for temporary data
OUTLOC = "./code/validation/pytest/encode"

random.seed(3)
np.random.seed(3)

###############################################################################
################################ HELPER FUNCS #################################
# Build a dummy namespace class to pass in to the check_args function
class DummyNamespace():
    
    def __init__(self, encoding, protein_name, fasta = None,
                positions = None, n_combined = None,
                output = os.getcwd(), batches = 0,
                batch_size = 4):
        
        self.encoding = encoding
        self.protein_name = protein_name
        self.fasta = fasta
        self.positions = positions
        self.n_combined = n_combined
        self.output = output
        self.batches = batches
        self.batch_size = batch_size
        
def insert_points(original_list, insertion_inds, character):
    """
    Inserts a single character into a series of positions in a list
    """
    # Sort insertion inds
    insertion_inds = sorted(insertion_inds)
    
    # Loop over indices where we will make insertions
    insertion_counter = 0
    for insertion_ind in insertion_inds:
        
        # Make insertion, taking into account that indices will shift after
        # each insertion
        original_list.insert(insertion_ind + insertion_counter, character)
        
        # Update the running dash ind. This considers how many insertions we have
        # already made 
        insertion_counter += 1
        
def finish_unprocessed_alignment(unprocessed_ref, n_aligned):
    
    # Populate a2m and a3m alignments with refseq
    full_alignment_a2m = []
    full_alignment_a3m = []
    processed_alignment = []
    
    joined_a2m = "".join(unprocessed_ref)
    joined_a3m = "".join(char for char in unprocessed_ref if char != ".")
    joined_processed = "".join(char for char in unprocessed_ref if char in PROCESSED_CHAR_OPTS)
    
    full_alignment_a2m.append(("Ref", joined_a2m))
    full_alignment_a3m.append(("Ref", joined_a3m))
    processed_alignment.append(("Ref", joined_processed))
    
    # Create a list for storing sequences we've already seen
    observed_a2m = [joined_a2m]
    observed_a3m = [joined_a3m]
    observed_processed = [joined_processed]
    
    # Build aligned sequences
    for alignment_ind in range(1, n_aligned + 1):

        # Create a candidate sequence
        candidate_seq = [None] * len(unprocessed_ref)
        for candidate_ind, unprocessed_char in enumerate(unprocessed_ref):

            # If the unprocessed character is a dash or uppercase character,
            # then we can choose either a dash or an uppercase character at random
            if unprocessed_char in PROCESSED_CHAR_OPTS:
                candidate_seq[candidate_ind] = random.choice(PROCESSED_CHAR_OPTS_LIST)

            # If the unprocessed character is a dot or lowercase character,
            # then we can choose either a lowercase character, a ., or a *
            elif unprocessed_char in UNPROCESSED_CHAR_OPTS:
                candidate_seq[candidate_ind] = random.choice(UNPROCESSED_CHAR_OPTS_LIST)

            # Anything else and we're doing it wrong
            else:
                raise AssertionError()

        # Add the candidate sequence on to the alignment if they are not already present
        joined_a2m = "".join(candidate_seq)
        if joined_a2m not in observed_a2m:
            observed_a2m.append(joined_a2m)
            full_alignment_a2m.append((f"Var{alignment_ind}", joined_a2m))
            
        joined_a3m = "".join(char for char in candidate_seq if char != ".")
        if joined_a3m not in observed_a3m:
            observed_a3m.append(joined_a3m)
            full_alignment_a3m.append((f"Var{alignment_ind}", joined_a3m))
            
        joined_processed = "".join(char for char in candidate_seq if char in PROCESSED_CHAR_OPTS)
        if joined_processed not in observed_processed:
            observed_processed.append(joined_processed)
            processed_alignment.append((f"Var{alignment_ind}", joined_processed))
        
    return full_alignment_a2m, full_alignment_a3m, processed_alignment
def generate_alignments():
    """
    Generator that prodcues randomly constructed alignments for stress-testing
    our functions
    """
    # Loop over alignment sizes
    for i in range(N_ALIGNMENT_TESTS):

        # Get the reference sequence length
        refseq_len = REFSEQ_LENGTHS[i]
        
        # Build the expected processed alignment
        og_seq = [random.choice(string.ascii_uppercase) for _ in range(REFSEQ_LENGTHS[i])]

        # Now convert elements of the original sequence at random to lowercase. 
        lowercase_positions = {ind for ind in range(refseq_len) 
                               if np.random.rand() < LOWERCASE_PROBS[i]}
        unprocessed_ref = [char.lower() if ind in lowercase_positions
                           else char for ind, char in enumerate(og_seq)]
        
        # At this point, choose some target positions. These should be 1-indexed and come
        # only from uppercase positions. These represent accessible positions for embedding.
        uppercase_positions = [ind for ind in range(refseq_len) if ind not in lowercase_positions]
        n_uppercase = len(uppercase_positions)
        n_targets = N_TARGET_POSITIONS[i] if n_uppercase >= N_TARGET_POSITIONS[i] else n_uppercase
        target_og_positions = np.random.choice(uppercase_positions, size = n_targets, replace = False) + 1

        # Insert random dashes
        dashpoints = np.random.randint(0, refseq_len, 
                                       size = int(DASH_PROBS[i] * refseq_len))
        insert_points(unprocessed_ref, dashpoints, "-")

        # Insert random dots
        new_size = len(unprocessed_ref)
        dotpoints = dashpoints = np.random.randint(0, new_size, size = int(DOT_PROBS[i] * new_size))
        insert_points(unprocessed_ref, dotpoints, ".")

        # Insert random stars
        new_size = len(unprocessed_ref)
        starpoints = dashpoints = np.random.randint(0, new_size, size = int(DOT_PROBS[i] * new_size))
        insert_points(unprocessed_ref, starpoints, "*")

        # Finally, build the remaining sequences
        alignments = finish_unprocessed_alignment(unprocessed_ref, ALIGNMENT_SIZES[i])
        
        yield ("".join(og_seq), lowercase_positions, target_og_positions, *alignments)
        
        
################################################################################
################################### TESTS ######################################

def test_check_args():
    
    # Loop over all encodings
    for encoding in ALL_ENCODINGS:

        # If learned, go this route
        if encoding in LEARNED_ENCODINGS:

            # Confirm that errors are thrown if 'fasta' or 'positions'
            # are not provided
            with pytest.raises(AssertionError, match = "'fasta' a required argument for learned embeddings"):
                check_args(DummyNamespace(encoding, "TEST", positions = ["T1"]))
            with pytest.raises(AssertionError, match = "'positions' are required for learned embeddings"):
                check_args(DummyNamespace(encoding, "TEST", fasta = "./TEST"))

        # If passing in a non-learned encoding, confirm that errors
        # are thrown if `n_combined` is not provided
        if encoding in REGULAR_ENCODINGS:
            with pytest.raises(AssertionError, match = f"'n_combined' a required argument for {encoding}"):
                check_args(DummyNamespace(encoding, "TEST", fasta = "./TEST"))

        # Confirm that setting a negative or 0 batch size raises an error
        with pytest.raises(AssertionError, match = "Batch size must be greater than or equal to 1"):
            check_args(DummyNamespace(encoding, "TEST", batch_size = 0))
        with pytest.raises(AssertionError, match = "Batch size must be greater than or equal to 1"):
            check_args(DummyNamespace(encoding, "TEST", batch_size = -1))

    # Confirm that encodings are not accepted
    for _ in range(100):
        with pytest.raises(AssertionError, match = "'encoding' must be one of"):
            dummy_encoding = "".join(random.choice(string.printable) for _ in range(30))
            check_args(DummyNamespace(dummy_encoding, "TEST"))
            
# Check that we are correctly processing alignments
def test_alignment_handling():
    """
    Tests all functions to do with loading and processing alignments
    """
    # Run tests on random alignments
    for og_seq, lowercase_positions, target_og, a2m, a3m, expected_processed in generate_alignments():

        # Make sure the expected processed matches that output by process alignments
        processed_a2m = process_alignment(a2m, DELETEKEYS)
        processed_a3m = process_alignment(a3m, DELETEKEYS)
        assert processed_a2m == expected_processed
        assert processed_a3m == expected_processed

        # Get the expected dictionary from both the a2m and a3m refseqs
        a2m_old_to_new = build_old_to_new(a2m[0][1], DELETEKEYS)
        a3m_old_to_new = build_old_to_new(a3m[0][1], DELETEKEYS)

        # Make sure that the two dictionaries are equal
        assert a2m_old_to_new == a3m_old_to_new

        # Make sure that the lowercase positions are not keys in the two dictionaries
        dict_keys = set(a2m_old_to_new.keys())
        assert len(lowercase_positions.intersection(dict_keys)) == 0
        assert len(dict_keys.intersection(lowercase_positions)) == 0

        # The dictionary should map from the original protein sequence to the processed
        # protein sequence
        processed_ref = processed_a2m[0][1]
        assert all(og_seq[key] == processed_ref[val] for key, val in a2m_old_to_new.items())

        # Translate the target positions to the processed alignment and make sure that 
        # these match the expected characters
        translated_targets = translate_target_positions(target_og, a2m_old_to_new)
        assert all(og_seq[og_pos - 1] == processed_ref[translated_pos - 1]
                  for og_pos, translated_pos in zip(target_og, translated_targets))

        # Test load_alignment for both alignment types
        for alignment, name in zip((a2m, a3m), ("a2m", "a3m")):

            # Save the alignments to disk
            alignment_savename = os.path.join(OUTLOC, f"TEMP.{name}")
            with open(alignment_savename, "w") as output_handle:
                sequences = [SeqRecord(Seq(seq), id = desc, description = "") for desc, seq in alignment]
                SeqIO.write(sequences, output_handle, "fasta")

            # Now load the alignment using the MLDE function
            loaded_processed, loaded_position_converter = load_alignment(alignment_savename)

            # Make sure the processed from the load function matches the expected processed
            # Also make sure the dictionary matches what we built in the other functions
            assert loaded_processed == expected_processed
            assert loaded_position_converter == a2m_old_to_new
            
            # Delete saved file
            os.remove(alignment_savename)