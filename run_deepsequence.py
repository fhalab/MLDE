"""
This initates a script that trains a DeepSequence model and then uses it for
making zero-shot predictions.
"""
def main():
    
    # Import relevant modules
    import argparse
    import os
    from deep_sequence.support_funcs import check_inputs
    
    # Instantiate argparser
    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument("alignment", help = ".a2m file used to train DeepSequence")
    parser.add_argument("--positions", help = "AA indices to target", required = True,
                        nargs = "+", dest = "positions", default = None, type = str)
    parser.add_argument("--output", help = "Save location for output predictions",
                        required = False, default = os.getcwd())
    parser.add_argument("--save_model", action = "store_true",
                        help = "Set to save the trained DeepSequence model parameters")
    parser.add_argument("--no_cudnn", action = "store_true",
                        help = "Set to turn off use of CUDNN libraries")
    
    # Parse the arguments and check
    args = parser.parse_args()
    check_inputs(args)
    
    # Now we need to set relevant environment variables for theano
    theano_flags = 'floatX=float32,device=cuda'
    if args.no_cudnn:
        theano_flags += ',dnn.enabled=False'
    os.environ['THEANO_FLAGS'] = theano_flags
    
    # Run DeepSequence and save results
    from deep_sequence.run_funcs import run_deepseq
    deep_seq_results = run_deepseq(args.alignment, args.positions,
                                   save_model = args.save_model)
    deep_seq_results.to_csv(os.path.join(args.output, "DeepSeqPreds.csv"),
                            index = False)
    
# Run as a script
if __name__ == "__main__":
    main()