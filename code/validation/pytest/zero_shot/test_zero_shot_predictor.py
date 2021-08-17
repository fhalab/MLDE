"""
This tests what we can with the zero-shot results. Note that because we don't
have a ground truth for the mask-filling protocols, we can't really do anything
to be sure on its effectiveness (aside from the fact that it works for zero-shot).
Abundant assertions and other checks during calculation give us the best idea
for its accuracy.
"""
# Import pytest and other modules needed for testing
import pytest
from ....zero_shot.zero_shot_predictor import ZeroShotPredictor

# Import other required modules
from ....encode.molbio_info import ALL_AAS
from evcouplings.couplings import CouplingsModel
import random

# Get a location for an evmutation model, a fasta sequence, and an alignment
FASTA_LOC = "./code/validation/basic_test_data/2GI9.fasta"
ALIGNMENT_LOC = "./code/validation/basic_test_data/GB1_Alignment.a2m"
EVMUT_LOC = "./code/validation/basic_test_data/GB1_EVcouplingsModel.model"

# Define test positions
TEST_POSITIONS = ("V39", "D40", "G41", "V54")

# Define the number of tests
N_TESTS = 320000

def test_predict_evmutation():

    # Build a zero shot predictor for evmutation and make predictions
    predictor = ZeroShotPredictor(FASTA_LOC, TEST_POSITIONS)
    evmut_preds = predictor.predict_evmutation(EVMUT_LOC)
    
    # Load in our own version of the predictor
    manual_predictor = CouplingsModel(EVMUT_LOC)

    # Loop over the number of tests
    for test in range(N_TESTS):

        # Make a random mutation
        random_combo = "".join(random.choice(ALL_AAS) for _ in range(len(TEST_POSITIONS)))

        # Build a mutation list
        mutlist = [(39, "V", random_combo[0]),
                   (40, "D", random_combo[1]),
                   (41, "G", random_combo[2]),
                   (54, "V", random_combo[3])]

        # Calculate
        test_pred, _, _ = manual_predictor.delta_hamiltonian(mutlist)

        # Pull the actual number from the prediction results
        true_res = evmut_preds.loc[evmut_preds.Combo.values == random_combo, "EvMutation"].values[0]

        # Make sure the results match
        assert test_pred == true_res