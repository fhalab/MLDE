"""
Runs zero-shot predictions using either EVmutation, DeepSequence, or any of
the mask-filling protocols with the ESM models and ProtBert models.
"""
def main():
    
    # Import necessary modules and functions
    import argparse
    import os

    # Import custom objects
    from code.zero_shot.support_funcs import check_args, run_zero_shot
    
    # Instantiate argparser
    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument("--positions", help = "AA indices to target", required = True,
                        nargs = "+", dest = "positions", default = None, type = str)
    parser.add_argument("--models", help = "The models to use for zero-shot prediction.",
                        required = True, nargs = "+", dest = "models", default = None,
                        type = str)
    parser.add_argument("--fasta", required = False, default = None,
                        help = "FASTA file containing parent sequence in .fasta format")
    parser.add_argument("--alignment", required = False, default = None,
                        help = ".a2m file used with MSATransformer")
    parser.add_argument("--evmutation_model", required = False, default = None,
                        help = "Path to model for use with EVmutation predictions")
    parser.add_argument("--include_conditional", action = "store_true",
                        help = "Set to also return calculations using conditional probability")
    parser.add_argument("--mask_col", action = "store_true", 
                        help = "Masks the full column of an MSA during prediction if using the MSA transformer")
    parser.add_argument("--batch_size", help = "Batch size for ESM and ProtBert calculations",
                        required = False, type = int, default = 4)
    parser.add_argument("--output", help = "Save location for output files.",
                        required = False, default = os.getcwd())

    # Parse the arguments
    args = parser.parse_args()

    # Check arguments
    check_args(args)

    # Now run zero-shot prediction
    run_zero_shot(args)
    
# Run only as a script
if __name__ == "__main__":
    main()