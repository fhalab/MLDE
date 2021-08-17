"""
This file contains all functions that support running DeepSequence
"""
# Import relevant modules
import os
import re
from Bio import SeqIO

# First we need a function that confirms the proposed positions are acceptable 
# for the proposed alignment
def check_inputs(args):
    """
    Checks the arguments input to `run_deepsequence.py` for validity. 
    
    Parameters
    ----------
    args: NameSpace: Namespace output by parsing the ArgumentParser in 
        `run_deepsequence.py`
    """
    # Define a set of allowed characters for a protein sequence. These are the
    # canonical amino acids
    allowed_aas = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                   "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
    
    # Unpack args
    alignment_loc = args.alignment
    input_positions = args.positions
    output_loc = args.output
    
    # Make sure the output location is real
    assert os.path.exists(output_loc), "Cannot find output location"

    # Load the reference sequence. This is the first position in the alignment
    assert os.path.exists(alignment_loc), "Cannot find alignment file"
    refseq = str(next(SeqIO.parse(alignment_loc, "fasta")).seq)

    # Check to make sure the input positions were made correctly
    input_splitter = re.compile("^([A-Z])([0-9]+)$")
    split_input_positions = [input_splitter.match(input_position) for
                             input_position in input_positions]
    if any(match_obj == None for match_obj in split_input_positions):
        raise AssertionError("Incorrectly formatted input positions. Format "
                             "should be AA# separated by spaces for each position. "
                             "AA should be capitalized.")

    # Get the expected aas and positions in the wildtype sequence
    expected_aas_positions = tuple((match_obj.group(1), int(match_obj.group(2)) - 1)
                                   for match_obj in split_input_positions)
    assert all(aa in allowed_aas for aa, _ in expected_aas_positions), "At least one input aa is not allowed"

    # Confirm that all positions are in range
    max_ind = len(refseq) - 1
    if not all((pos >= 0) and (pos <= max_ind) for _, pos in expected_aas_positions):
        raise AssertionError("Amino acid indices must be in the range 1 to {}".format(max_ind + 1))

    # Confirm that the alignment has these sequences and that positions are in order
    mismatches = []
    previous_pos = -1
    for i, (expected_aa, expected_pos) in enumerate(expected_aas_positions):
        found_aa = refseq[expected_pos]
        if found_aa != expected_aa:
            mismatches.append("Expected {} but found {}".format(input_positions[i], 
                                                               "{}{}".format(found_aa, expected_pos + 1)))
        assert previous_pos < expected_pos, "Out of order input"
        previous_pos = expected_pos

    # Report any mismatches
    if len(mismatches) > 0:
        print("Please fix the following errors in requested positions:")
        for mismatch in mismatches:
            print(mismatch)
        raise AssertionError("Mismatch between refseq and requested positions")