"""
This module contains a class needed to generate all encoding types.
"""
# Import necessary MLDE modules
from .MolBioInfo import all_aas, allowed_aas
from .GeorgievParams import georgiev_parameters
from .TapeModelLocations import tape_model_locations

# Import other necessary modules
import os
import re
import warnings
import pickle
import subprocess
import numpy as np
from itertools import product
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from time import strftime

# Write a regex that splits protein amino acid indices into amino acid code and
# integer
_aa_ind_splitter = re.compile("([A-Za-z])([0-9]+)")

# Get the directory of this file
_filedir = os.path.dirname(os.path.abspath(__file__))

#===============================================================================
#============================== Helper Functions ===============================
#===============================================================================
# Write a function that normalizes encodings
def _normalize_encodings(unnormalized_encodings):
    """
    Takes a tensor of embeddings, flattens the internal dimensions, then mean-centers
    and unit-scales. After scaling, the matrix is repacked to its original shape.
    
    Parameters
    ----------
    unnormalized_encodings: Numpy array of shape N x A x D, where N is the number
        of combinations in the design space, A is the number of amino acids in 
        the combination, and D is the dimensionality of the base encoding. This
        array contains the unnormalized MLDE encodings.
        
    Returns
    -------
    normalized_encodings: Numpy array with the same shape as the input, only
        with encodings mean-centered and unit-scaled
    """
    # Raise an error if the input array is not 3D
    assert len(unnormalized_encodings.shape) == 3, "Input array must be 3D"
    
    # Get the length of a flattened array
    flat_shape = np.prod(unnormalized_encodings.shape[1:])

    # Flatten the embeddings along the internal axes
    flat_encodings = np.reshape(unnormalized_encodings,
                                 [len(unnormalized_encodings), flat_shape])

    # Mean-center and unit variance scale the embeddings.
    means = flat_encodings.mean(axis=0)
    stds = flat_encodings.std(axis=0)
    normalized_flat_encodings = (flat_encodings - means)/stds
    
    # Reshape the normalized embeddings back to the original form
    normalized_encodings = np.reshape(normalized_flat_encodings,
                                      unnormalized_encodings.shape)
    
    return normalized_encodings

# Define a class for generating embeddings
class EncodingGenerator():
    """
    The class which contains all information needed to generate encodings.
    
    Parameters
    ----------
    encoding: str: Choice of "learned", "georgiev", and "onehot". This dictates
        how the combinatorial space will be encoded.
    protein_name: str: Nickname for the combinatorial space that will be built.
    fasta_path: str (default = None): Path to the fasta file containing the parent
        protein sequence. This argument is required when using learned embeddings;
        it is ignored when using other encodings.
    target_protein_indices: list of str: Positions in the protein to encode. Must
        take the format 'WTaaPos', where 'WTaa' is the wild-type amino acid at
        position 'Pos' (1-indexed): e.g. V20 means that position 20 in the protein
        given by 'fasta_path' has Valine in the wild type, and encodings should be
        built for it. A list of positions defines the combinatorial space to encode.
        This argument is required when using learned embeddings; otherwise it is
        ignored.
    n_positions_combined: int: The number of amino acids to combine. Ignored 
        when using learned embeddings, required otherwise.
    output: str: Location to save all data. By default, this is the current
        working directory.
    
    Returns
    -------
    None. Outputs are saved to the location given by 'output'.
    """
    # Initialize the embedding
    def __init__(self, encoding, protein_name,
                 fasta_path = None, target_protein_indices = None, 
                 n_positions_combined = None, output = os.getcwd()):

        # Assign all inputs as instance variables
        self._encoding = encoding.lower()
        self._fasta_path = fasta_path
        self._target_protein_indices = target_protein_indices
        self._protein_name = protein_name
        self._output = output
        
        # Additional checks if working with learned embeddings. 
        if encoding in {"resnet", "bepler", "unirep", "transformer", "lstm"}:
            
            # Assert that we have the correct variables present
            assert target_protein_indices is not None, "Did not define target indices"
            assert fasta_path is not None, "Did not specify location of fasta file"
            
            # Load the fasta file
            self._process_input_fasta()

            # Check the input indices
            self._check_indices()
        
        elif encoding in {"georgiev", "onehot"}:
            
            # Assert that we have the correct variables present
            assert n_positions_combined is not None, "Did not define n_positions_combined"
            
            # Assign a combinatorial space size and get a count of the variant size
            self._n_positions_combined = n_positions_combined
        
        else:
            raise AssertionError("Unknown encoding")
        
        # Define the size of the combinatorial space
        self._combi_space = 20**self.n_positions_combined
        
        # Build output directories
        self._build_output_dirs()
        
        # Build the list of combinations for the position and the dictionaries 
        # linking position index to combo
        self._build_combo_dicts()

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

        # Load the fasta file
        with open(self.fasta_path, "r") as f:
            
            # Open the fasta file and extract all sequences
            fasta_seqs = list(SeqIO.parse(f, "fasta"))
            
        # Assert that there is only one sequence in fasta_seqs
        assert len(fasta_seqs) == 1, "Embedding generator can currently only handle 1 parent sequence"

        # Convert the full sequence to uppercase and record
        wt_prot_seq = str(fasta_seqs[0].seq).upper()

        # Confirm that only allowed characters are present in the input sequence
        forbidden_characters = [[i, char] for i, char in enumerate(wt_prot_seq)
                                if char not in allowed_aas]

        # If we have any forbidden characters, report them
        if len(forbidden_characters) > 0:
            print("Please fix errors at the below protein sequence indices:")
            for i, char in forbidden_characters:
                print("Forbidden character: {} at position {}".format(char, i+1))
            raise AssertionError("Forbidden character in input sequence.")

        # If we are otherwise okay, set the protein sequence as an instance variable
        self._wt_seq = wt_prot_seq

    # Define a function for confirming that the indices we have selected are okay
    def _check_indices(self):
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
        index_checks = [_aa_ind_splitter.match(index) for
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
        mismatches = []
        for i, (letter, ind) in enumerate(aa_letter_index):

            # Pull the associated letter
            expected_letter = self.wt_seq[ind]

            # If the letter doesn't match, report
            if expected_letter != letter:
                mismatches.append("You requested {}, but found {}{}".
                                   format(self.target_protein_indices[i],
                                          expected_letter, ind + 1))

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
        self._wt_aas = [aa for aa, _ in aa_letter_index]
        self._target_python_inds = [ind for _, ind in aa_letter_index]

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
            
    # Write a function that builds output directories
    def _build_output_dirs(self):
        """
        Self-explanatory: Build necessary directories for saving data.
        """
        # Get the start time
        init_time = strftime("%Y%m%d-%H%M%S")
        
        # Build the output directories consistent for all encodings
        self._encoding_output = os.path.join(self.output, init_time, "Encodings")
        os.makedirs(self.encoding_output)
        
        # Build the output directories only used only for generating learned embeddings
        self._fasta_output = os.path.join(self.output, init_time, "Fastas")
        os.makedirs(self.fasta_output)
            
    # Write a function that produces dictionaries linking combo and index in the
    # output encodings
    def _build_combo_dicts(self):
        """
        Builds dictionaries which link the identity of a combination (e.g. ACTV)
        to that combination's index in an encoding array, and vice versa. Both
        dictionaries are saved to disk.
        """
        # Identify all possible combinations
        self._all_combos = list(product(all_aas, repeat = self.n_positions_combined))
        
        # Link combo to index and index to combo
        combo_to_index = {"".join(combo): i for i, combo in enumerate(self.all_combos)}
        self._index_to_combo = {i: "".join(combo) for i, combo in enumerate(self.all_combos)}
        
        # Save the constructed dictionaries
        with open(os.path.join(self.encoding_output, f"{self.protein_name}_{self.encoding}_ComboToIndex.pkl"), "wb") as f:
            pickle.dump(combo_to_index, f)
        with open(os.path.join(self.encoding_output, f"{self.protein_name}_{self.encoding}_IndexToCombo.pkl"), "wb") as f:
            pickle.dump(self.index_to_combo, f)
        
    # Write a function that generates fasta files for the requested variants
    def _build_fastas(self, n_batches):
        """
        The TAPE program requiers a fasta file as an input. It will then encode
        all proteins in the input fasta file. This function takes the input
        fasta file and builds a new fasta file containing all possible variants
        in the combinatorial space. To save on system RAM, this task is split into
        n_batches number of batches.
        """
        # Convert the wt sequence to a list
        wt_list = list(self.wt_seq)

        # Create a list to store all fasta filenames in
        fasta_filenames = [None for _ in range(n_batches)]

        # Loop over the number of batches
        for i, combo_batch in enumerate(np.array_split(self.all_combos, n_batches)):

            # Create a filename for the file we will use to store fasta data
            fasta_filename = os.path.join(self.fasta_output,
                                          "{}_Batch{}_Variants.fasta".format(self.protein_name, i))

            # Record the fasta_filename
            fasta_filenames[i] = fasta_filename
            
            # Create a list in which we will store SeqRecords
            temp_seqs = [None for _ in range(len(combo_batch))]
            
            # Build fasta for the batch
            for j, combo in enumerate(combo_batch):

                # Make a copy of the wild type list
                temp_seq = wt_list.copy()

                # Create a list to store the variant name
                var_name = [None for _ in range(self.n_positions_combined)]

                # Loop over the target python indices and set new amino acids
                for k, (aa, ind) in enumerate(zip(combo, self.target_python_inds)):

                    # Replace the WT amino acids with the new ones
                    temp_seq[ind] = aa

                    # Format the variant name (OldaaIndNewaa)
                    var_name[k] = "{}{}{}".format(self.wt_aas[k], ind + 1, aa)

                # Create and store a SeqRecords object
                variant_name = f"{self.protein_name}_{'-'.join(var_name)}"
                temp_seqs[j] = SeqRecord(Seq("".join(temp_seq)),
                                         id = variant_name,  description="")

            # Write fasta sequences of the variant combinations to the file
            with open(fasta_filename, "w") as f:
                SeqIO.write(temp_seqs, f, "fasta")

        # Return the filename of the fasta sequences
        return fasta_filenames

    # Write a function that generates onehot encodings
    def _generate_onehot(self):
        """
        Builds a onehot encoding for a given combinatorial space.
        """
        # Make a dictionary that links amino acid to index
        one_hot_dict = {aa: i for i, aa in enumerate(all_aas)}
    
        # Build an array of zeros
        onehot_array = np.zeros([len(self.all_combos), self.n_positions_combined, 20])
        
        # Loop over all combos. This should all be vectorized at some point.
        for i, combo in enumerate(self.all_combos):
            
            # Loop over the combo and add ones as appropriate
            for j, character in enumerate(combo):
                
                # Add a 1 to the appropriate position
                onehot_ind = one_hot_dict[character]
                onehot_array[i, j, onehot_ind] = 1
                
        # Return the array
        return onehot_array
        
    # Write a function that generates georgiev encodings
    def _generate_georgiev(self):
        """
        Encodes a given combinatorial space with Georgiev parameters.
        """
        # Now build all encodings for every combination
        unnorm_encodings = np.empty([len(self.all_combos), 
                                       self.n_positions_combined, 19])
        for i, combo in enumerate(self.all_combos):
            unnorm_encodings[i] = [[georgiev_param[character] for georgiev_param
                                    in georgiev_parameters] for character in combo]
            
        return unnorm_encodings
        
    # Write a function that generates learned encodings
    def _generate_learned(self, n_batches):
        """
        Encodes a given combinatorial space using tape.
        Unlike Georgiev and one-hot encodings, these encodings are context-
        aware. To save on system RAM, this task is split into n_batches number
        of batches.
        """
                # Build fasta files
        fasta_filenames = self._build_fastas(n_batches)

        # Create a list to store the names of the raw embedding files
        extracted_embeddings = [None for _ in range(n_batches)]

        # Get the name of the model and the load-from name
        weight_loc = tape_model_locations[self.encoding]
        
        # Get a temporary filename for storing batches
        temp_filename = os.path.join(_filedir, "Buffer", "TempOutputs.pkl")

        # Loop over the number of batches
        for i, fasta_filename in tqdm(enumerate(fasta_filenames),
                                      desc = "Batch#", total = len(fasta_filenames),
                                      position = 0):

            # Run TAPE to get the transformer embeddings
            _ = subprocess.run(["tape-embed", fasta_filename,
                                self.encoding, "--load-from", weight_loc,
                                "--output", temp_filename])

            # Load the raw embedding file that was generated
            with open(temp_filename, "rb") as f:
                raw_embeddings = pickle.load(f)
            
            # Extract just the indices we care about
            extracted_embeddings[i] = np.array([protein_embedding[0, self.target_python_inds, :]
                                                for protein_embedding in raw_embeddings])
        
        # Delete the temporary outputs file
        os.remove(temp_filename)
        
        # Return the extracted embeddings, concatenating along the first dimension
        return np.concatenate(extracted_embeddings)
               
    #===========================================================================
    #============================== Public Methods =============================
    #===========================================================================    
    # Write a function that generates encodings
    def generate_encodings(self, n_batches = 1):
        """
        Generates encodings based on the self.encoding instance variable of the
        encoding. Also performs KS-sampling if desired. Note that this class is
        not currently set up to be reused. In other words, a new class should
        be instantiated for generating a new set of encodings. 
        
        Parameters
        ----------
        n_batches: int: The number of batches to split the job into. TAPE heavily
            uses system RAM, and splitting into batches lowers the memory 
            requirements.
            
        Returns
        -------
        None. Will save normalized and unnormalized encodings to disk. If KS-
            sampling is performed, this will be saved to disk as well.
        """
        # Generate the appropriate encoding
        if self.encoding in {"resnet", "bepler", "unirep", "transformer", "lstm"}:
            unnormalized_embeddings = self._generate_learned(n_batches)
        elif self.encoding == "georgiev":
            unnormalized_embeddings = self._generate_georgiev()
        elif self.encoding == "onehot":
            
            # Get the embeddings
            onehot_array = self._generate_onehot()
            
            # Save the encodings
            savename = os.path.join(self.encoding_output,
                                    f"{self.protein_name}_onehot_UnNormalized.npy")
            np.save(savename, onehot_array)
            
            # Return
            return None
            
        else:
            raise ValueError(f"Unknown encoding type {self.encoding}")
        
        # Normalize embeddings
        # Reshape the normalized embeddings back to the original form
        normalized_embeddings = _normalize_encodings(unnormalized_embeddings)

        # Create filenames for saving the embeddings
        unnormalized_savename = os.path.join(self.encoding_output, 
                                             f"{self.protein_name}_{self.encoding}_UnNormalized.npy")
        norm_savename = os.path.join(self.encoding_output,
                                     f"{self.protein_name}_{self.encoding}_Normalized.npy")

        # Save the embeddings
        np.save(unnormalized_savename, unnormalized_embeddings)
        np.save(norm_savename, normalized_embeddings)
        
        
    # =========================================================================
    # ============== Protect instance variables as attributes =================
    # =========================================================================
    @property
    def encoding(self):
        return self._encoding

    @property
    def fasta_path(self):
        return self._fasta_path
        
    @property
    def target_protein_indices(self):
        return self._target_protein_indices

    @property
    def protein_name(self):
        return self._protein_name

    @property
    def output(self):
        return self._output

    @property
    def n_positions_combined(self):
        return self._n_positions_combined

    @property
    def combi_space(self):
        return self._combi_space

    @property
    def wt_seq(self):
        return self._wt_seq

    @property
    def wt_aas(self):
        return self._wt_aas

    @property
    def target_python_inds(self):
        return self._target_python_inds

    @property
    def encoding_output(self):
        return self._encoding_output

    @property
    def fasta_output(self):
        return self._fasta_output
    
    @property
    def all_combos(self):
        return self._all_combos
    
    @property
    def index_to_combo(self):
        return self._index_to_combo