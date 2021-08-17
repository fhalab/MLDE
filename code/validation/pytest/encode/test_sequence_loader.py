# Load pytest and the Sequence loader
import pytest
from ....encode.sequence_loader import SequenceLoader

# Load other required modules
import os
from Bio import SeqIO

# Define globals that will be used in many checks
test_data_loc = "./code/validation/pytest/encode/test_data/"
FASTA_PATHS = [os.path.join(test_data_loc, "DummyFasta.fasta")]
MSA_PATHS = [os.path.join(test_data_loc, "DummyA2M.a2m"),
             os.path.join(test_data_loc, "DummyA3M.a3m")]
TARGET_PROTEIN_INDS = ["Y4", "M11", "L20"]
EXPECTED_WT_AAS = ("Y", "M", "L")
MSA_TARGET_INDS = [1, 7, 15]
TARGET_PYTHON_INDS = [3, 10, 19]
N_POSITIONS_COMBINED = len(TARGET_PROTEIN_INDS)
TEST_PROCESSED_ALIGNMENT = [(seq.description, str(seq.seq)) for seq in
                            SeqIO.parse(os.path.join(test_data_loc, "DummyProcessedAlignment.a2m"), "fasta")]
TEST_REFSEQ = str(next(SeqIO.parse(os.path.join(test_data_loc, "DummyFasta.fasta"), "fasta")).seq)
EXPECTED_PROCESSED_REFSEQ = TEST_PROCESSED_ALIGNMENT[0][1]

# Define the forbidden character information
FORBIDDEN_CHARACTER_MSAS = [os.path.join(test_data_loc, "DummyA2MForbiddenCharacters.a2m"),
                            os.path.join(test_data_loc, "DummyA3MForbiddenCharacters.a3m")]
FORBIDDEN_CHARACTER_FASTA = [os.path.join(test_data_loc, "DummyFastaForbiddenCharacters.fasta")]

# Test the basics of the initialization of the object
def test_init():

    # Make tests that apply for both the transformers and msa
    # transformer
    for msa_check in (True, False):

        # Get the expected parent sequence
        expected_par = TEST_PROCESSED_ALIGNMENT if msa_check else TEST_REFSEQ

        # Loop over the fasta paths
        test_ins = MSA_PATHS if msa_check else FASTA_PATHS
        for fasta_path in test_ins:

            # Make sure the checks on indices and the fasta path 
            # are active
            with pytest.raises(AssertionError, match = "Did not define target indices"):
                SequenceLoader(fasta_path = fasta_path, msa_transformer = msa_check)
            with pytest.raises(AssertionError, match = "Did not specify location of fasta file"):
                SequenceLoader(target_protein_indices = TARGET_PROTEIN_INDS,
                               msa_transformer = msa_check)

            # For a correct load, make sure that all properties come back correct
            correct_loader = SequenceLoader(fasta_path = fasta_path,
                                            target_protein_indices = TARGET_PROTEIN_INDS,
                                            msa_transformer = msa_check)
            assert correct_loader.fasta_path == fasta_path
            assert correct_loader.msa_transformer == msa_check
            assert correct_loader.target_protein_indices == TARGET_PROTEIN_INDS
            assert correct_loader.n_positions_combined == N_POSITIONS_COMBINED
            assert correct_loader.target_python_inds == TARGET_PYTHON_INDS
            assert correct_loader.wt_seq == expected_par
            assert correct_loader.wt_aas == EXPECTED_WT_AAS    
            
# Test the fasta processing function
def test_process_input_fasta():
    
    # Checks that apply to both the msa transformer and the regular
    # transformers
    for msa_check in (True, False):

        # Pass in a bad fasta location and make sure an error is thrown
        bad_loc = "./Bad.fasta"
        with pytest.raises(IOError, match = f"Cannot locate '{bad_loc}'"):
            SequenceLoader(fasta_path = bad_loc,
                           target_protein_indices = TARGET_PROTEIN_INDS,
                           msa_transformer = msa_check)

        # Loop over the fasta paths
        test_ins = FORBIDDEN_CHARACTER_MSAS if msa_check else FORBIDDEN_CHARACTER_FASTA
        for fasta_path in test_ins:

            # If a forbidden character is passed in, throw an error
            with pytest.raises(AssertionError, match = "Forbidden character in input sequence."):
                SequenceLoader(fasta_path = fasta_path,
                              target_protein_indices = TARGET_PROTEIN_INDS,
                              msa_transformer = msa_check)

    # An error should be thrown passing a fasta file in to an MSA transformer
    with pytest.raises(AssertionError, match = "Expected alignment, but received fasta"):
        SequenceLoader(fasta_path = FASTA_PATHS[0],
                       target_protein_indices = TARGET_PROTEIN_INDS,
                       msa_transformer = True)

    # Make sure the output sequences are correct for a non-msa
    good_nonmsa = SequenceLoader(fasta_path = FASTA_PATHS[0],
                                 target_protein_indices = TARGET_PROTEIN_INDS,
                                 msa_transformer = False)
    wt_prot_seq, pos_converter = good_nonmsa._process_input_fasta()
    assert wt_prot_seq == TEST_REFSEQ
    assert pos_converter == None

    # Checks that apply just to either of the msa transformer inputs
    for msa_file in MSA_PATHS:

        # If we pass in an alignment file to a non-msa transformer, an 
        # error should be thrown
        with pytest.raises(AssertionError, match = "Embedding generator can currently only handle 1 parent sequence"):
            SequenceLoader(fasta_path = msa_file,
                           target_protein_indices = TARGET_PROTEIN_INDS,
                           msa_transformer = False)

        # Make sure the output sequences are correct for an msa
        msa_loader = SequenceLoader(fasta_path = msa_file,
                                     target_protein_indices = TARGET_PROTEIN_INDS,
                                     msa_transformer = True)
        wt_prot_seq, pos_converter = msa_loader._process_input_fasta()
        assert wt_prot_seq == EXPECTED_PROCESSED_REFSEQ
        assert isinstance(pos_converter, dict)
        
def test_check_indices():
    
    # Define a set of improperly formatted input indices to test
    bad_formats = [["4Y", "M11", "L20"],
                ["Y4", "M 11", "L20"],
                ["Y4", "M11", "L20G"],
                [" Y4", "M11", "L20"],
                ["Y4 ", "M11", "L20"],
                ["YY4", "M11", "L20"]]

    # Build a set of mismatched mutations
    bad_mutations = [["Y4", "M7", "L15"],
                    ["M4", "M11", "L20"],
                    ["Y4", "L11", "L20"],
                     ["Y4", "M11", "Y20"],
                     ["F4", "M11", "L20"],
                     ["Y4", "A11", "L20"],
                     ["Y4", "M11", "G20"]]

    # Build a set of out of range mutations for the msa
    # transformer. This includes positions that go away
    # after processing
    out_of_range_msa = [["V1", "Y4"],
                        ["A2", "Y4"],
                        ["F3", "Y4"],
                       ["Q5", "Y4"],
                       ["Y4", "E27"]]

    # Build a set of out of range mutations for a regular transformer
    out_of_range_regular = [["V0", "Y4"],
                           ["Y4", "Y29"]]

    # Build a set of duplicate mutations
    duplicates = [["Y4", "Y4", "M11", "L20"],
                 ["Y4", "M11", "M11", "L20"],
                 ["Y4", "M11", "L20", "L20"]]

    # Build a set of out of order mutations
    out_of_order = [["M11", "Y4", "L20"],
                    ["M11", "L20", "Y4"],
                   ["L20", "M11", "Y4"],
                   ["L20", "Y4", "M11"],
                   ["Y4", "L20", "M11"],
                   ["Y4", "M11", "Y4"]]

    # Checks that apply to both the msa transformer and the regular
    # transformers
    for msa_check in (True, False):

        # Loop over the fasta paths
        test_ins = MSA_PATHS if msa_check else FASTA_PATHS
        for fasta_path in test_ins:

            # Confirm that improper formatting will raise an error
            for bad_input in bad_formats:
                with pytest.raises(AssertionError, match = "Unrecognizable protein index."):
                    SequenceLoader(fasta_path = fasta_path,
                                  target_protein_indices = bad_input,
                                  msa_transformer = msa_check)

            # Throw error if the mutation can't be found at the 
            # requested position
            for bad_input in bad_mutations:
                with pytest.raises(AssertionError, match = "Requested positions not found."):
                    SequenceLoader(fasta_path = fasta_path,
                                   target_protein_indices = bad_input,
                                   msa_transformer = msa_check)

            # Compile a list of out of range mutations
            if msa_check:
                oor_muts = out_of_range_msa + out_of_range_regular
            else:
                oor_muts = out_of_range_regular

            # Make sure all out of range mutations raise an error
            for bad_input in oor_muts:
                with pytest.raises(AssertionError, match = "Out of range AA index"):
                    SequenceLoader(fasta_path = fasta_path,
                                   target_protein_indices = bad_input,
                                   msa_transformer = msa_check)

            # Throw an error if we have duplicate indices
            for bad_input in duplicates:
                with pytest.raises(AssertionError, match = "Duplicate indices identified."):
                    SequenceLoader(fasta_path = fasta_path,
                                   target_protein_indices = bad_input,
                                   msa_transformer = msa_check)

            # Throw an error if indices are out of order
            for bad_input in out_of_order:
                with pytest.raises(AssertionError, match = "Out of order indices."):
                    SequenceLoader(fasta_path = fasta_path,
                                   target_protein_indices = bad_input,
                                   msa_transformer = msa_check)