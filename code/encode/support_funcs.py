# Load psutil
import psutil
import string
import numpy as np
from Bio import SeqIO

# Load custom modules/objects
from .model_info import N_LATENT_DIMS

# Define a function which checks EncodingGenerator 
def check_args(parsed_args):
    """
    Checks the parsed arguments input to GenerateEncodings.py. 
    """
    # Make sure that the encoding is allowed
    allowed_encodings = {"resnet", "bepler", "unirep", "transformer", "lstm", 
                         "georgiev", "onehot", "esm1b_t33_650M_UR50S", 
                         "esm1_t34_670M_UR50S", "esm1_t34_670M_UR50D",
                         "esm1_t34_670M_UR100", "esm1_t12_85M_UR50S",
                         "esm1_t6_43M_UR50S", "esm_msa1_t12_100M_UR50S",
                         "prot_bert_bfd", "prot_bert"}
    assert parsed_args.encoding in allowed_encodings, f"'encoding' must be one of {allowed_encodings}"

    # Batch size must be greater than or equal to 1
    assert parsed_args.batch_size >= 1, "Batch size must be greater than or equal to 1"

    # learned requires fasta_path and target_protein_indices. Make sure this is present
    if parsed_args.encoding in N_LATENT_DIMS:
        assert parsed_args.fasta is not None, "'fasta' a required argument for learned embeddings"
        assert parsed_args.positions is not None, "'positions' are required for learned embeddings"
        
    # onehot and georgiev require n_positions_combined. Make sure this is present.
    else:
        assert parsed_args.n_combined is not None, f"'n_combined' a required argument for {parsed_args.encoding}"
    
# Write a function that calculates batch size
def calculate_batch_size(embedding_obj):
    """
    Calculates the recommended number of batches based on the size of the
    target combinatorial space.
    """    
    # Get the maximum available RAM. The program is configured to never use
    # more than 75% of the available system RAM
    mem_obj = psutil.virtual_memory()
    total_available = mem_obj.available * 0.75
    
    # Get the total required RAM. The total required RAM is calibrated from
    # the approximate required RAM used to generate a 4-site combinatorial
    # library in GB1 (which needs 64 GB of RAM in one batch for a 512 latent-dimension)
    # model. The constant is given by 64 * 1024**3 / (56 * 4 * 20**4) / 512 ~= 4 bytes/aa/latent_dim
    RAM_constant =  4
    total_RAM_needed = (RAM_constant * embedding_obj.combi_space * \
                        N_LATENT_DIMS[embedding_obj.encoding] \
                        * len(embedding_obj.wt_seq) * embedding_obj.n_positions_combined)

    # Calculate the number of recommended batches. This is the floor divide of
    # the total RAM needed and the maximum available RAM
    n_batches = int(np.ceil(total_RAM_needed / total_available))

    # If n_batches was set to 0, up to 1 batch
    if n_batches == 0:
        n_batches = 1
        
    # Return the number of batches
    return n_batches

def process_alignment(unprocessed_alignment, deletekeys):
    """
    This handles the input alignments to the MSA transformer. Specifically, it 
    reformats the alignment such that all unaligned columns are eliminated and
    duplicate sequences are deleted. Unaligned columns are those with "." and
    lowercase letters. The example code provided in ESM also omits the "*"
    character (see 
    https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb),
    so this character is also ommitted here for consistency. Note that, because
    a3m is just an a2m file format with all "." symbols removed (see page 26 of 
    the HHSuite docs: 
    http://sysbio.rnet.missouri.edu/bdm_download/DeepRank_db_tools/tools/DNCON2/hhsuite-2.0.16-linux-x86_64/hhsuite-userguide.pdf
    this conversion should handle both a2m and a3m files and convert them to the
    same output. This file 
    
    Parameters
    ----------
    unprocessed_alignment: list of lists: An unprocessed a2m or a3m alignment
        file formatted such that each entry is (description, sequence).
    deketekeys: dict: The keys to delete from all sequences in the unprocessed
        alignment. This includes all lowercase characters, ".", and "*". The
        format is {character: None} for each character to delete.
            
    Returns
    -------
    processed_alignment: list of lists: An a2m or a3m alignment file with all
        unaligned columns and duplicate sequences removed.
    """ 
    # Create the translation table
    translation = str.maketrans(deletekeys)
    
    # Loop over elements of the unprocessed alignment
    processed_alignment = []
    observed_seqs = []
    for desc, seq in unprocessed_alignment:

        # Translate and add to the processed alignment if it has
        # not previously been observed
        processed_seq = seq.translate(translation)
        if processed_seq not in observed_seqs:
            observed_seqs.append(processed_seq)
            processed_alignment.append((desc, processed_seq))
            
    # Confirm that all sequences are the same length
    testlen = len(processed_alignment[0][1])
    assert all(len(seq) == testlen for _, seq in processed_alignment)
    
    return processed_alignment

def build_old_to_new(unprocessed_refseq, deletekeys):
    """
    Processing an alignment with `process_alignment` changes the indices of the
    mutated positions relative to their original locations in the unprocessed
    sequence. This function builds a dictionary that relates the old index (in
    the unprocessed alignment) to the new index (in the processed alignment).
    
    Parameters
    ----------
    unprocessed_refseq: str: The first sequence in the unprocessed alignment. 
    deletekeys: dict: The keys to delete from all sequences in the unprocessed
        alignment. This includes all lowercase characters, ".", and "*". The
        format is {character: None} for each character to delete.
        
    Returns
    -------
    old_to_new_pos: dict: A dictionary that relates the old index in the reference
        sequence (!! 0-indexed !!) to the new position in the processed 
        reference sequence (!! also 0-indexed !!).
    """
    # Build a dictionary linking the old positions in the protein to
    # the new. Note that this dictionary is 0-indexed relative to the
    # protein sequence
    # Get the number of alphabetic characters in the reference
    n_capital_letters = sum((char.isalpha() and char.isupper()) 
                            for char in unprocessed_refseq)

    # Loop over each character in the unprocessed reference sequence
    seq_ind = -1
    processed_ind = -1
    old_to_new_pos = {}
    for char in unprocessed_refseq:
        
        # Check if the character is a letter and whether or not it is
        # in the deletekeys
        alpha_check = char.isalpha()
        delete_check = (char not in deletekeys)
        
        # If the character is a letter, increment the sequence index. Letters
        # are the only characters that match the original sequence
        if alpha_check:
            seq_ind += 1
            
        # If the character is not in the set of deletekeys, increment the
        # processed index. Characters not in the deletekeys are carried into
        # the processed sequences
        if delete_check:
            
            # Increment counter
            processed_ind += 1
            
            # Sanity check: If not a letter, then this must be "-"
            if not alpha_check:
                assert char == "-", "Unexpected character in reference sequence"
        
        # If the character is both alphabetic and not in the deletekeys, then
        # record it as a viable character that can be converted
        if alpha_check and delete_check:
            old_to_new_pos[seq_ind] = processed_ind 
            
    # Confirm that we captured all sequence elements that we could
    assert len(old_to_new_pos) == n_capital_letters
                
    return old_to_new_pos

def load_alignment(input_filename):
    """
    Given the path to an alignment file, loads the alignment, then processes it
    to remove unaligned columns. The processed alignment is then ready to be 
    passed to the tokenization function of the MsaTransformer.
    
    Parameters
    ----------
    input_filename: str: Path to the alignment. 
    
    Returns
    -------
    processed_alignment: list of lists: Contents of an a2m or a3m alignment file
        with all unaligned columns removed. This is formatted for passage into
        the tokenization function of the MsaTransformer.
    old_to_new_pos: dict: A dictionary that relates the old index in the reference
        sequence to the new position in the processed reference sequence.
    """
    # Set up deletekeys. This code is taken right from ESM
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    
    # Load the unprocessed alignment
    unprocessed_alignment = [(record.description, str(record.seq))
                             for record in SeqIO.parse(input_filename, "fasta")]

    # Save the original reference sequence
    unprocessed_refseq = unprocessed_alignment[0][1]

    # Get a dictionary linking old position to processed position
    position_converter = build_old_to_new(unprocessed_refseq, deletekeys)

    # Process the alignment
    processed_alignment = process_alignment(unprocessed_alignment, deletekeys)
    
    # We only need the processed alignment and the dictionary of old to new
    return processed_alignment, position_converter

def translate_target_positions(target_positions, position_converter):
    """
    Converts the original positions in a protein (!! 1-indexed relative to the
    sequence !!) to the associated positions in a processed alignment file. The
    returned positions are 1-indexed relative to the alignment. The reason for
    1-indexing relative to the sequence is because the returned tokens and 
    representations have a <cls> token added to them, meaning that something
    1-indexed relative to the sequence is 0-indexed relative to the representation.

    Parameters
    ----------
    target_positions: 1d numpy array: The positions to mutate in a protein. These
        are 1-indexed relative to the protein sequence, 0-indexed relative to
        output representations and the tokenized sequence.
    position_converter: dict: Output of `build_old_to_new`. Note that this 
        dictionary converts 0-indexed indices in the sequence to 0-indexed indices
        in the processed sequence. Thus, "1" must be subtracted from the 
        `target_positions` array before it is converted using this converter. 
        We then add "1" back to the translated seqs after conversion to get them
        back to 1-indexing relative to the processed sequence.
        
    Returns
    -------
    translated_targets: 1d numpy array: The positions to mutate in the processed
        protein. These are 1-indexed relative to the protein sequence, 0-indexed
        relative to output representations and the tokenized sequence.
    """
    # Our input target positions are 1-indexed relative to the sequence. The
    # first thing we need to do is convert them to zero-indexed positions.
    zero_indexed_targets = target_positions - 1
    
    # The position converter is 0-indexed relative to the sequence. Now that the
    # target positions have been translated to a 0-index, we can translate them
    # to their new position in the protein
    translated_targets = np.array([position_converter[target] for target in zero_indexed_targets])
    
    # Finally, we want the positions to be 1-indexed relative to the sequence
    # so that they are 0-indexed relative to the tokenization (which has a cls token
    # prepended, shifting indexing for the targets)
    return translated_targets + 1

# Write a function that normalizes encodings
def normalize_encodings(unnormalized_encodings):
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