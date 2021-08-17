"""
This file contains an object which wraps all major zero-shot strategies.
"""
# Load 3rd party modules
import os
import itertools
import numpy as np
import pandas as pd
from evcouplings.couplings import CouplingsModel

# Load custom modules
from ..encode.sequence_loader import SequenceLoader
from ..encode.molbio_info import ALL_AAS
from ..encode.transformer_classes import TRANSFORMER_TO_CLASS


class ZeroShotPredictor(SequenceLoader):

    def __init__(self, fasta_path, target_protein_indices, msa_transformer = False):

        # Initialize the parent module if not using 
        super().__init__(fasta_path = fasta_path,
                         target_protein_indices = target_protein_indices,
                         msa_transformer = msa_transformer)
    
        # Define a stationary list of all possible combos
        self._all_combos = list(itertools.product(ALL_AAS, repeat = self.n_positions_combined))
        
    def predict_evmutation(self, model_path):
        
        # Load a model
        model = CouplingsModel(model_path)

        # Define an array for storing outputs
        preds = np.empty(len(self.all_combos))

        # Loop over all combos
        for i, combo in enumerate(self.all_combos):

            # If the combo is wt, result is 0 and we can continue on
            # to the next
            if combo == self.wt_aas:
                preds[i] = 0
                continue

            # Build the mutation list. Don't include mutations that go back to wt
            mutation_list = [(self.target_positions[j], self.wt_aas[j], mut_char)
                            for j, mut_char in enumerate(combo)
                            if self.wt_aas[j] != mut_char]

            # Make the prediction
            preds[i], _, _ = model.delta_hamiltonian(mutation_list)

        # Create a dataframe that we can then return
        return self.preds_to_df(preds, "EvMutation")
    
    # Now make a function for predicting using ESM models:
    def predict_esm(self, model_name, batch_size = 4, naive = True, full_col = False):

        # First make sure the model exists
        assert model_name in TRANSFORMER_TO_CLASS, "Unknown model"

        # Then load up the appropriate class
        model = TRANSFORMER_TO_CLASS[model_name](model_name)

        # Now make predictions
        preds = model.zero_shot_pred(self.wt_seq, self.target_positions,
                                     self.all_combos, self.wt_aas, 
                                     batch_size = batch_size, naive = naive,
                                     full_col = full_col)

        # Get the column name for the result
        suffix = "Naive" if naive else "Conditional"
        report_name = f"{model_name}-{suffix}"
        if self.msa_transformer:
            suffix = "ColumnMasked" if full_col else "ColumnUnmasked"
            report_name = f"{report_name}-{suffix}"
        
        return self.preds_to_df(preds, report_name)
            
    def preds_to_df(self, preds, score_name):    
        
        # Build and return a dataframe of predictions
        return pd.DataFrame({"Combo": ["".join(combo) for combo in self.all_combos],
                             score_name: preds})
    
    @property
    def all_combos(self):
        return self._all_combos