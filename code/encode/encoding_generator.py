"""
This module contains a class needed to generate all encoding types.
"""
# Import 3rd party modules
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

# Import necessary MLDE modules
from .molbio_info import ALL_AAS, ALLOWED_AAS
from .georgiev_params import GEORGIEV_PARAMETERS
from .support_funcs import normalize_encodings
from .model_info import TAPE_MODEL_LOCATIONS, N_LATENT_DIMS, TRANSFORMER_INFO
from .sequence_loader import SequenceLoader

# Try and import the TransformerToClass module. This is needed when using ESM
# and ProtBert for encoding and relies on having pytorch present in the Python
# environment. It will thus not always be loadable depending on the environment
# used -- this is an optional import
try:
    from .transformer_classes import TRANSFORMER_TO_CLASS
except ModuleNotFoundError:
    warnings.warn("Could not load `TRANSFORMER_TO_CLASS`. This is expected if "
                  "you are running in the `mlde` environment or a custom environment "
                  "without PyTorch installed. Otherwise, you might have a problem. "
                  "It will not be possible to build encodings from ESM and ProtTrans "
                  "with this import not working.")


# Get the directory of this file
_FILEDIR = os.path.dirname(os.path.abspath(__file__))

# Define a class for generating embeddings
class EncodingGenerator(SequenceLoader):
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

        # We need to load in the requested sequence if working with learned
        # embeddings
        if encoding in N_LATENT_DIMS:
            
            # Record whether or not this is an msa transformer
            msa_transformer = True if encoding == "esm_msa1_t12_100M_UR50S" else False
            
            # Activate the sequence loader if we are using encodings/embeddings. 
            super().__init__(fasta_path = fasta_path,
                             target_protein_indices = target_protein_indices,
                             msa_transformer = msa_transformer)
        
        # Otherwise we just set essential variables for hard-coded encodings
        elif encoding in {"georgiev", "onehot"}:
            
            # Assert that we have the correct variables present
            assert n_positions_combined is not None, "Did not define n_positions_combined"
            
            # Assign a combinatorial space size and get a count of the variant size
            self._n_positions_combined = n_positions_combined
        
        # If we have requested an unsupported encoding, raise an error
        else:
            raise AssertionError("Unknown encoding")
        
        # Assign all inputs shared by all states as instance variables
        self._encoding = encoding
        self._protein_name = protein_name
        self._output = output            

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
        self._all_combos = list(product(ALL_AAS, repeat = self.n_positions_combined))
        
        # Link combo to index and index to combo
        combo_to_index = {"".join(combo): i for i, combo in enumerate(self.all_combos)}
        index_to_combo = {i: "".join(combo) for i, combo in enumerate(self.all_combos)}
        
        # Save the constructed dictionaries
        with open(os.path.join(self.encoding_output, f"{self.protein_name}_{self.encoding}_ComboToIndex.pkl"), "wb") as f:
            pickle.dump(combo_to_index, f)
        with open(os.path.join(self.encoding_output, f"{self.protein_name}_{self.encoding}_IndexToCombo.pkl"), "wb") as f:
            pickle.dump(index_to_combo, f)
        
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
        one_hot_dict = {aa: i for i, aa in enumerate(ALL_AAS)}
    
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
                                    in GEORGIEV_PARAMETERS] for character in combo]
            
        return unnorm_encodings
        
    # Write a function that generates learned encodings
    def _generate_tape(self, n_batches):
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
        weight_loc = TAPE_MODEL_LOCATIONS[self.encoding]
        
        # Get a temporary filename for storing batches
        temp_filename = os.path.join(self.encoding_output,
                                     f"{self.protein_name}_{self.encoding}_TempOutputs.pkl")

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
    
    def _generate_transformer(self, n_batches, batch_size):
                
        # Instantiate the model
        model = TRANSFORMER_TO_CLASS[self.encoding](self.encoding)
        
        # Run encoding
        return model.encode_combinatorial_lib(self.wt_seq, self.target_positions,
                                              self.all_combos, self.wt_aas,
                                              batch_size = batch_size, 
                                              n_processing_batches = n_batches)
               
    #===========================================================================
    #============================== Public Methods =============================
    #===========================================================================    
    # Write a function that generates encodings
    def generate_encodings(self, n_batches = 1, batch_size = 4):
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
            unnormalized_embeddings = self._generate_tape(n_batches)
        
        elif self.encoding in TRANSFORMER_INFO:
            _, unnormalized_embeddings = self._generate_transformer(n_batches,
                                                                    batch_size)
            
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
        normalized_embeddings = normalize_encodings(unnormalized_embeddings)

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
    def protein_name(self):
        return self._protein_name

    @property
    def output(self):
        return self._output

    @property
    def combi_space(self):
        return self._combi_space

    @property
    def encoding_output(self):
        return self._encoding_output

    @property
    def fasta_output(self):
        return self._fasta_output
    
    @property
    def all_combos(self):
        return self._all_combos