"""
This file contains an abstract class for loading fasta and alignment files 
"""
# Import 3rd party modules
import os
import re
import warnings
import numpy as np
from Bio import SeqIO

# Import custom modules
from .support_funcs import load_alignment, translate_target_positions
from .molbio_info import ALLOWED_AAS

class SequenceLoader():
    """
    """
    # Initialize the embedding
    def __init__(self, fasta_path = None, target_protein_indices = None, 
                 msa_transformer = False):

        # Assert that we have the correct variables present
        assert target_protein_indices is not None, "Did not define target indices"
        assert fasta_path is not None, "Did not specify location of fasta file"

        # Assign all inputs as instance variables
        self._fasta_path = fasta_path
        self._target_protein_indices = target_protein_indices
        
        # Record whether or not this is an input sequence or an input msa tan msa transformer
        self._msa_transformer = msa_transformer
                                    
        # Load the fasta file
        wt_prot_seq, position_converter = self._process_input_fasta()

        # Check the input indices
        self._check_indices(wt_prot_seq, position_converter)
                
    #===========================================================================
    #============================ Private Methods ==============================
    #===========================================================================
    # Write a function that analyzes the input fasta file
    def _process_input_fasta(self):
        """
        Loads the input fasta file and makes sure it passes a number of checks.
        Sets the variable self._wt_seq, which contains the sequence in the input
        fasta.
        """
        # Check to make sure the file exists
        if not os.path.exists(self.fasta_path):
            raise IOError("Cannot locate '{}'".format(self.fasta_path))
            
        # If we are using the msa transformer, load the alignment
        if self.msa_transformer:
            
            # Load the alignment and the position converter dict
            processed_alignment, position_converter = load_alignment(self.fasta_path)
            assert len(processed_alignment) > 1, "Expected alignment, but received fasta"
            
            # Set the wildtype sequence as the processed alignment
            self._wt_seq = processed_alignment
            
            # Set the sequence to check as the first element of the alignment
            wt_prot_seq = processed_alignment[0][1]
            
        # Otherwise, we load the first sequence in the fasta file
        else:
                
            # Load the fasta file
            with open(self.fasta_path, "r") as f:
                
                # Open the fasta file and extract all sequences
                fasta_seqs = list(SeqIO.parse(f, "fasta"))
                
            # Assert that there is only one sequence in fasta_seqs
            assert len(fasta_seqs) == 1, "Embedding generator can currently only handle 1 parent sequence"

            # Convert the full sequence to uppercase and record
            wt_prot_seq = str(fasta_seqs[0].seq).upper()
            
            # Set the protein sequence as an instance variable
            self._wt_seq = wt_prot_seq
            
            # Create a dummy variable for the position converter
            position_converter = None

        # Confirm that only allowed characters are present in the input sequence.
        # Note that the msa transformer can have a gap in it's reference sequence.
        if self.msa_transformer:
            forbidden_characters = [[i, char] for i, char in enumerate(wt_prot_seq)
                                    if char not in ALLOWED_AAS and char != "-"]
        else:
            forbidden_characters = [[i, char] for i, char in enumerate(wt_prot_seq)
                                    if char not in ALLOWED_AAS]

        # If we have any forbidden characters, report them
        if len(forbidden_characters) > 0:
            print("Please fix errors at the below protein sequence indices:")
            for i, char in forbidden_characters:
                print("Forbidden character: {} at position {}".format(char, i+1))
            raise AssertionError("Forbidden character in input sequence.")
        
        # Return the sequences we want to check indices against as well as the
        # dictionary converting old protein positions into new
        return wt_prot_seq, position_converter

    # Define a function for confirming that the indices we have selected are okay
    def _check_indices(self, wt_prot_seq, position_converter):
        """
        Compares the input positions to mutate against the wild-type sequence to
        ensure that sites chosen are legitimate. Sets the instance variables
        self._n_positions_combined, self._wt_aas, and self._target_python_inds.
        """
        # Set the number of requested indices as an instance variable
        self._n_positions_combined = len(self.target_protein_indices)

        # If a user passes in more than 4 indices, warn them
        if self.n_positions_combined > 4:
            warnings.warn("Embedding more than 4 combinatorial sites is extremely resource intensive",
                          ResourceWarning)

            # Have the user explicitly accept moving forward
            while True:
                answer = input("Are you sure you want to continue?").lower()
                if answer=="yes":
                    break
                elif answer=="no":
                    quit()
                else:
                    print("Please answer 'yes' or 'no'.")

        # Split protein indices into amino acid code and number
        aa_ind_splitter = re.compile("^([A-Z])([0-9]+)$")
        index_checks = [aa_ind_splitter.match(index) for
                        index in self.target_protein_indices]

        # If any checks are "None", throw an error
        missed_matches = [self.target_protein_indices[i] for i, check in
                          enumerate(index_checks) if check is None]
        if len(missed_matches) > 0:
            print("Protein indices must take the form AA# (e.g. S102 for serine at position 102).")
            print("Please fix errors in the below input indices:")
            for missed_match in missed_matches:
                print(missed_match)
            raise AssertionError("Unrecognizable protein index.")

        # Gather the amino acid letter and python protein index from the match objects
        aa_letter_index = [[match.group(1), int(match.group(2)) - 1] for
                           match in index_checks]
        
        # Confirm that the amino acid letter at the index matches the
        # input protein sequence
        max_ind = len(wt_prot_seq) - 1
        mismatches = []
        for i, (letter, req_ind) in enumerate(aa_letter_index):

            # Copy the originally requested index
            ind = req_ind

            # Convert the index if this is an msa transformer
            if self.msa_transformer:
                
                # Raise an execption if this index cannot be found
                assert ind in position_converter, "Out of range AA index"
                    
                # Convert the index
                ind = position_converter[ind]

            # Confirm that the index is in range
            assert (ind >= 0) and (ind <= max_ind), "Out of range AA index"

            # Pull the associated letter
            expected_letter = wt_prot_seq[ind]

            # If the letter doesn't match, report
            if expected_letter != letter:
                mismatches.append("You requested {}, but found {}{}".
                                   format(self.target_protein_indices[i],
                                          expected_letter, req_ind + 1))

        # Report any mismatches between the input sequence and requested indices
        if len(mismatches) > 0:
            print("There is a mismatch between your requested combinatorial positions and the input fasta sequence.")
            print("Remember that this program treats protein sequences as 1-indexed.")
            print("The below mismatches were identified between your requested positions and the input sequence:")
            for mismatch in mismatches:
                print(mismatch)
            raise AssertionError("Requested positions not found.")

        # If we've made it through all of the checks, record the wild-type
        # amino acids at the requested positions as well as the python indices
        self._wt_aas = tuple([aa for aa, _ in aa_letter_index])
        self._target_python_inds = [ind for _, ind in aa_letter_index]
        
        # Set 1-indexed indices. These are the original indices of the protein
        # where we want to make mutations
        self._target_positions = np.array(self.target_python_inds) + 1
        
        # If using the msatransformer, update the target positions
        if self.msa_transformer:
            self._target_positions = translate_target_positions(self.target_positions,
                                                                position_converter)

        # One last check to make sure that the indices are in order and that we
        # don't have any duplicates
        last_ind = -1
        for ind in self.target_python_inds:
            
            # Make sure there are no duplicate indices
            if ind == last_ind:
                print("All entered amino acid positions must be unique.")
                raise AssertionError("Duplicate indices identified.")


            # Check to make sure that the index is later than the previous
            elif ind < last_ind:
                print("Do us all a favor and enter the amino acid indices in order...")
                print("You'll thank yourself later.")
                raise AssertionError("Out of order indices.")
            
            # Reassign the last ind
            last_ind = ind
            
    @property
    def fasta_path(self):
        return self._fasta_path
    
    @property
    def msa_transformer(self):
        return self._msa_transformer
    
    @property
    def target_protein_indices(self):
        return self._target_protein_indices
    
    @property
    def n_positions_combined(self):
        return self._n_positions_combined
    
    @property
    def target_python_inds(self):
        return self._target_python_inds
    
    @property
    def target_positions(self):
        return self._target_positions
    
    @property
    def wt_seq(self):
        return self._wt_seq
    
    @property
    def wt_aas(self):
        return self._wt_aas