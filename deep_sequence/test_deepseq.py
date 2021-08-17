# Load pytest and the functions to test
import pytest
from .support_funcs import check_inputs
from .run_funcs import build_mutant_file, predict_mutants
from .globals import DEEPSEQ_WORKING_DIR
from DeepSequence import model, helper

# Import random number generators
import random
import string
import math
import os
import pandas as pd
import re
from collections import Counter
from Bio import SeqIO
from scipy.stats import spearmanr

# Decide on the number of tests
N_TESTS = 50

# Define the allowed AAs
ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
ALLOWED_AAS = set(ALL_AAS)

# Define the positions that we can mutate over
MAX_ALLOWED_POSITIONS = 1500
ALLOWED_POSITIONS = list(range(MAX_ALLOWED_POSITIONS))

# Define the allowed number of elements in the mutation library
MAX_N_ELEMENTS = 4

# Define a regex for splitting mutations
MUT_MATCHER = re.compile("^([A-Z])([0-9]+)([A-Z])$")

# Define the number of iterations to run for VAE preds
ITER_OVERRIDE = 10

def test_check_inputs():
    
    # Create default arguments
    default_align = "./code/validation/pytest/encode/test_data/DummyA2M.a2m"
    default_positions = ("S6", "L10", "G19")
    default_output = "./"
    default_save_model = False
    default_no_cudnn = False
    
    # Create a dummy argument class
    class DummyNamespace():
        
        def __init__(self, alignment = default_align,
                     positions = default_positions, output = default_output,
                     save_model = default_save_model, no_cudnn = default_no_cudnn):
            
            self.alignment = alignment
            self.positions = positions
            self.output = output
            self.save_model = save_model
            self.no_cudnn = no_cudnn
            
    # Get the length of the reference sequence
    refseq_len = len(str(next(SeqIO.parse(default_align, "fasta")).seq))
    refseq_pos_list = list(range(1, refseq_len + 1))
            
    # Make sure we fail if the output location does not exist
    with pytest.raises(AssertionError, match="Cannot find output location"):
        check_inputs(DummyNamespace(output = "./adklfjhajkebhclkajdnhfhaje")) # Points if you have this
        
    # Make sure we fail if the alignment can't be found
    with pytest.raises(AssertionError, match="Cannot find alignment file"):
        check_inputs(DummyNamespace(alignment= "./adklfjhajkebhclkajdnhfhaje.fasta"))
        
    # Test a series of badly formatted inputs. They should all fail.
    bad_ins = (("6S", "10L", "19G"),
               ("S6", "10L", "19G"),
               ("S_6", "L_10", "G_19"),
               ("S6L10", "G19"),
               ("S6", "L10G19"),
               ("S6L10G19",))
    for bad_in in bad_ins:
        with pytest.raises(AssertionError, match="Incorrectly formatted input positions"):
            check_inputs(DummyNamespace(positions=bad_in))
    
    # Build a set of mismatched mutations and make sure they are recognized as such
    bad_mutations = [["Y4", "M7", "L15"],
                    ["M4", "M11", "L20"],
                    ["Y4", "L11", "L20"],
                     ["Y4", "M11", "Y20"],
                     ["F4", "M11", "L20"],
                     ["Y4", "A11", "L20"],
                     ["Y4", "M11", "F20"]]
    for bad_mutset in bad_mutations:
        with pytest.raises(AssertionError, match = "Mismatch between refseq and requested positions"):
            check_inputs(DummyNamespace(positions=bad_mutset))
    
    # Determine bad inputs. These are positions that are out of range and 
    # characters that aren't allowed amino acids
    out_of_range = list(range(refseq_len, MAX_ALLOWED_POSITIONS))
    unallowed_aas = {char for char in string.ascii_uppercase 
                     if char not in ALLOWED_AAS}
    
    # Perform a given number of tests
    for _ in range(N_TESTS):
        
        # Grab some unallowed AAs and feed them in to the program
        n_positions = random.randrange(1, MAX_N_ELEMENTS + 1)
        positions_ids = sorted(random.sample(refseq_pos_list, n_positions))
        bad_aas = random.sample(unallowed_aas, n_positions)
        positions = tuple(bad_aa + str(pos) for bad_aa, pos in zip(bad_aas, positions_ids))
        
        # Make sure we throw an error with bad aas
        with pytest.raises(AssertionError, match = "At least one input aa is not allowed"):
            check_inputs(DummyNamespace(positions = positions))
            
        # Get a set of bad positions
        bad_position_ids = sorted(random.sample(out_of_range, n_positions))
        acceptable_aas = random.sample(list(ALLOWED_AAS), n_positions)
        oorange_positions = tuple(aa + str(bad_pos) for aa, bad_pos 
                                   in zip(acceptable_aas, bad_position_ids))
        
        # Make sure we throw an error with positions out of range
        with pytest.raises(AssertionError, match = "Amino acid indices must be in the range 1 to"):
            check_inputs(DummyNamespace(positions=oorange_positions))
            
        # Make sure we throw an error with unsorted positions
        bad_sort_positions = tuple(aa + str(bad_pos) for aa, bad_pos in 
                                   zip(acceptable_aas, reversed(positions_ids)))
        if n_positions > 1:    
            with pytest.raises(AssertionError, match = "Out of order input"):
                check_inputs(DummyNamespace(positions=bad_sort_positions))

def test_build_mutant_file():
    
    # Loop over the given number of tests
    for _ in range(N_TESTS):

        # Decide on (1) the number of random positions, (2) what those
        # positions are, and (3) what the WT is at those positions
        n_positions = random.randrange(1, MAX_N_ELEMENTS + 1)
        positions_ids = sorted(random.sample(ALLOWED_POSITIONS, n_positions))
        wt_ids = [random.choice(ALL_AAS) for _ in range(n_positions)]
        n_combos = 20 ** n_positions

        # Now build the inputs for the function
        mut_builder_inputs = tuple(str(wt_id) + str(position_id) for wt_id, position_id in zip(wt_ids, positions_ids))

        # Run the mutant file generation function
        test_file_loc, test_detail_to_short = build_mutant_file(mut_builder_inputs)

        # The detail to short should always contain the number of combinations minus 1 (as
        # we don't include the wt)
        assert (n_combos - 1) == len(test_detail_to_short)

        # Confirm that all shorthands are unique
        assert len(set(test_detail_to_short.values())) == len(test_detail_to_short.values())

        # Now load the test file
        test_file = pd.read_csv(os.path.join(DEEPSEQ_WORKING_DIR, test_file_loc))

        # Loop over the test file and confirm that all positions are represented.
        # The wt should never be present.
        mutation_counts = Counter()
        for mutant_list in test_file.itertuples(index = False):

            # Split the mutant on the delimiter
            split_muts = mutant_list.mutant.split(":")

            # Get the length and record the numbers
            n_muts = len(split_muts)
            assert n_muts > 0
            mutation_counts.update([n_muts])

            # Get the shorthand
            expected_shorthand = test_detail_to_short[mutant_list.mutant]

            # Break into the component parts
            by_element_muts = [MUT_MATCHER.match(split_mut).groups() for split_mut in split_muts]

            # Get the positions captured
            positions_captured = [int(by_element_mut[1]) for by_element_mut in by_element_muts]

            # Confirm that the positions are correct. The WT should match the first element of 
            # the mutation list unless the mutation is also the wt
            actual_shorthand = [None] * n_positions
            for i, pos in enumerate(positions_ids):

                # Get the expected wt
                expected_wt = wt_ids[i]

                # If not a captured positon, record the wt and continue
                if pos not in positions_captured:
                    actual_shorthand[i] = expected_wt
                    continue

                # Get the index of the position in the by_elements_muts
                by_element_ind = positions_captured.index(int(pos))

                # Get the observed wt and mut
                observed_wt = by_element_muts[by_element_ind][0]
                observed_mut = by_element_muts[by_element_ind][2]

                # Make sure that the wt does not equal the mutant
                assert expected_wt != observed_mut

                # Make sure that we have the expected wt
                assert expected_wt == observed_wt

                # Record the shorthand
                actual_shorthand[i] = observed_mut

            # Make sure the shorthand matches
            assert "".join(actual_shorthand) == expected_shorthand

        # Make sure we have the appropriate number of counts
        for i in range(n_muts):

            # Get the number of ways we can make the combination
            n_choices = i + 1
            n_ways = math.factorial(n_positions) / (math.factorial(n_choices) * (math.factorial(n_positions - n_choices)))

            # Get the expected number of counts and make sure that's what we found
            expected_counts = (19 ** n_choices) * n_ways
            assert expected_counts == mutation_counts[n_choices]

def test_predict_mutants():
        
    # Load in elements needed for testing predict_mutants
    test_datahelper = helper.DataHelper(alignment_file = "./deep_sequence/DeepSequence/examples/alignments/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m",
                                        working_dir = DEEPSEQ_WORKING_DIR, calc_weights = False)
    
    # Load the model
    model_params = {
            "batch_size"        :   100,
            "encode_dim_zero"   :   1500,
            "encode_dim_one"    :   1500,
            "decode_dim_zero"   :   100,
            "decode_dim_one"    :   500,
            "n_patterns"        :   4,
            "n_latent"          :   30,
            "logit_p"           :   0.001,
            "sparsity"          :   "logit",
            "encode_nonlin"     :   "relu",
            "decode_nonlin"     :   "relu",
            "final_decode_nonlin":  "sigmoid",
            "output_bias"       :   True,
            "final_pwm_scale"   :   True,
            "conv_pat"          :   True,
            "d_c_size"          :   40
            }

    test_vae_model   = model.VariationalAutoencoder(test_datahelper,
        batch_size              =   model_params["batch_size"],
        encoder_architecture    =   [model_params["encode_dim_zero"],
                                    model_params["encode_dim_one"]],
        decoder_architecture    =   [model_params["decode_dim_zero"],
                                    model_params["decode_dim_one"]],
        n_latent                =   model_params["n_latent"],
        n_patterns              =   model_params["n_patterns"],
        convolve_patterns       =   model_params["conv_pat"],
        conv_decoder_size       =   model_params["d_c_size"],
        logit_p                 =   model_params["logit_p"],
        sparsity                =   model_params["sparsity"],
        encode_nonlinearity_type       =   model_params["encode_nonlin"],
        decode_nonlinearity_type       =   model_params["decode_nonlin"],
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        output_bias             =   model_params["output_bias"],
        final_pwm_scale         =   model_params["final_pwm_scale"],
        working_dir             =   DEEPSEQ_WORKING_DIR)
        
    # Load model parameters
    test_vae_model.load_parameters(file_prefix = "PABP_YEAST")
    
    # Load mutations
    mutlist = pd.read_csv("./deep_sequence/DeepSequence/examples/mutations/PABP_YEAST_Fields2013-singles.csv").mutant.tolist()

    # Make double mutants and save the file
    double_muts = [mut1 + ":" + mut2 for mut1, mut2 in zip(mutlist, reversed(mutlist))]
    pd.DataFrame({"mutant": double_muts}).to_csv("./deep_sequence/DeepSequence/examples/mutations/val_muts.csv",
                                                index = False)

    # Make a combo to shorthand dict
    combo_to_short = {mut: str(i) for i, mut in enumerate(double_muts)}

    # Predict mutations using the deepsequence inbuilts
    mut_name_list, delta_elbos = test_datahelper.custom_mutant_matrix("mutations/val_muts.csv",
                                                                    test_vae_model, 
                                                                    N_pred_iterations=ITER_OVERRIDE)

    # Prediction mutations using the predict function
    mlde_preds = predict_mutants(test_datahelper, test_vae_model, ("G126", "P200"),
                                "mutations/val_muts.csv", combo_to_short, 
                                _include_assert = False, _iter_override = ITER_OVERRIDE)

    # The last entry in the predictions should be WT
    assert mlde_preds.Combo.tolist()[-1] == "GP"
    assert mlde_preds.DeepSequence.tolist()[-1] == 0

    # Confirm that the length of the mlde preds is 1 greater than the delta elbos
    assert len(mlde_preds) - len(delta_elbos) == 1

    # Confirm that we can recreate the mut name list
    converted_name_list = [combo_to_short[mut] for mut in mut_name_list]
    assert converted_name_list == mlde_preds.Combo.tolist()[:-1]

    # Make sure the spearman correlation between the two sets of predictions is above 
    # a threshold. They won't be exact due to seeding differences
    assert spearmanr(mlde_preds.DeepSequence.values[:-1], delta_elbos) > 0.9

    # Test some bad entries.
    with pytest.raises(AssertionError, match = "Missing calculation for DeepSeq. Maybe missing positions in the input alignment?"):
        _ = predict_mutants(test_datahelper, test_vae_model, ("G126", "P200"),
                            "mutations/val_muts.csv", {}, _iter_override = ITER_OVERRIDE)
        
    with pytest.raises(AssertionError, match = "Unexpected number of predictions"):
        _ = predict_mutants(test_datahelper, test_vae_model, ("G126", "P200"),
                            "mutations/val_muts.csv", combo_to_short, _iter_override = ITER_OVERRIDE)
    
        
    