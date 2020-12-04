# Load psutil
import psutil
import numpy as np

# Define a function which checks EncodingGenerator 
def check_args(parsed_args):
    """
    Checks the parsed arguments input to GenerateEncodings.py. 
    """
    # Make sure that the encoding is allowed
    allowed_encodings = {"resnet", "bepler", "unirep", "transformer", "lstm", 
                         "georgiev", "onehot"}
    assert parsed_args.encoding in allowed_encodings, f"'encoding' must be one of {allowed_encodings}"

    # learned requires fasta_path and target_protein_indices. Make sure this is present
    if parsed_args.encoding in {"resnet", "bepler", "unirep", "transformer", "lstm"}:
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
    
    # Get the number of latent dimensions
    n_latent_dims = {"resnet": 256, 
                     "bepler": 100, 
                     "unirep": 1900,
                     "transformer": 512,
                     "lstm": 2048}

    # Get the total required RAM. The total required RAM is calibrated from
    # the approximate required RAM used to generate a 4-site combinatorial
    # library in GB1 (which needs 64 GB of RAM in one batch for a 512 latent-dimension)
    # model. The constant is given by 64 * 1024**3 / (56 * 4 * 20**4) / 512 ~= 4 bytes/aa/latent_dim
    RAM_constant =  4
    total_RAM_needed = (RAM_constant * embedding_obj.combi_space * \
                        n_latent_dims[embedding_obj.encoding] \
                        * len(embedding_obj.wt_seq) * embedding_obj.n_positions_combined)

    # Calculate the number of recommended batches. This is the floor divide of
    # the total RAM needed and the maximum available RAM
    n_batches = int(np.ceil(total_RAM_needed / total_available))

    # If n_batches was set to 0, up to 1 batch
    if n_batches == 0:
        n_batches = 1
        
    # Return the number of batches
    return n_batches