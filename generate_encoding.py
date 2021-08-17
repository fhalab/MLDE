"""
This script generates encodings for a combination of positions. Options are to
encode as one-hot, georgiev, or learned embeddings. 
"""
# Define main(). Everyhing will be encompassed in this function
def main():
    
    # Import necessary modules and functions
    import argparse
    import os
    
    # Turn off extensive tensorflow readout
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    # Import MLDE functions and classes
    from code.encode.encoding_generator import EncodingGenerator
    from code.encode.support_funcs import calculate_batch_size, check_args
    from code.encode.model_info import N_LATENT_DIMS

    # Instantiate argparser
    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument("encoding", help = "Model to use for encoding")
    parser.add_argument("protein_name", help = "Protein name alias")
    parser.add_argument("--fasta", help = "FASTA file containing parent sequence in .fasta foramt or else alignment in .a2m or .a3m format", 
                        required = False, default = None, type = str)
    parser.add_argument("--positions", help = "AA indices to target",
                        required = False, nargs = "+", dest = "positions", default = None, type = str)
    parser.add_argument("--n_combined", help = "Number of positions to combine",
                        required = False, default = None, type = int)
    parser.add_argument("--output", help = "Save location for output files.",
                        required = False, default = os.getcwd())
    parser.add_argument("--batches", help = "Number of batches for embedding calculations",
                        required = False, type = int, default = 0)
    parser.add_argument("--batch_size", help = "Batch size for ESM and ProtBert calculations",
                        required = False, type = int, default = 4)

    # Parse the arguments
    args = parser.parse_args()
    
    # Make sure the arguments are appropriate
    check_args(args)    
 
    # Construct the embedding generator
    embedding_obj = EncodingGenerator(args.encoding, args.protein_name, 
                                      fasta_path = args.fasta,
                                      target_protein_indices = args.positions,
                                      n_positions_combined = args.n_combined,
                                      output = args.output)

    # If not specified and we are working with learned embeddings, 
    # decide on the number of batches
    if args.batches <= 0 and args.encoding in N_LATENT_DIMS:
        n_batches = calculate_batch_size(embedding_obj)
    else:
        n_batches = args.batches

    # Generate embeddings
    embedding_obj.generate_encodings(n_batches = n_batches, 
                                     batch_size = args.batch_size)

# Execute if run as a script
if __name__=="__main__":
    main()