"""
This module contains classes that enable encoding and zero-shot predictions 
using the transformer models provided in the ESM package 
(https://github.com/facebookresearch/esm) as well as the ProtBert package
(https://github.com/agemagician/ProtTrans)
"""
# Import required modules
import string
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import itertools
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer, BertForMaskedLM

# Import custom objects
from .model_info import TRANSFORMER_INFO
from .molbio_info import ALL_AAS

################################################################################
############################## Abstract Classes ################################
################################################################################

class AbstractBertEncoder(ABC):
    """
    Abstract class that can be used as a template for the different BERT
    models in ESM and ProtBert.
    """
    def __init__(self, model_name):
        """
        Loads in the model given by `model_name` and defines the following 
        isntance variables:
        
        self.device: The device used for computation.
        self.model_name: The input model name
        self.encoding_dim: The dimensionality of the encodings output by the model
        self.encoding_layer: The layer to use for encoding (only relevant for ESM)
        self.token_dim: The shape of a single tokenized sequence. Should be 2 for
            everything except the MsaTransformer
        self.tok_to_idx: A dictionary converting a token string to its index
        self.mask_string: The string defining a <mask> token
        self.cls_string: The string defining a <cls> token
        self.eos_string: The string defining the last token in a sequence. This 
            is <sep> for the protbert models and <eos> for the ESM models.
        self.alphabet_size: The number of characters in a model's alphabet
        self.model: A pytorch instance of the model
        self.tokenization_shape: The expected shape of a single tokenized sequence
        
        Parameters
        ----------
        model_name: str: The name of the model to load.
        """
        # Set the device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Get info on the model
        self._model_name = model_name
        self._encoding_dim, self._encoding_layer, self._token_dim = TRANSFORMER_INFO[model_name]
            
        # Get info on the vocab        
        (self._tok_to_idx, self._mask_string,
         self._cls_string, self._eos_string) = self._load_model(model_name)
        self._alphabet_size = len(self.tok_to_idx)
        
        # Send the model to the appropriate device
        self.model.to(self.device)
        self.model.eval()
        
        # Placeholder
        self._tokenization_shape = None
                
    def encode_combinatorial_lib(self, sequence, target_positions, all_combos,
                                 parent_combo, batch_size = 4, n_processing_batches = 1):
        """
        Builds the encodings for a combinatorial library using a transformer 
        model.
        
        Parameters
        ----------
        sequence: str or list of lists: If using an MsaTransformer, this is a 
            processed alignment (see support_funcs.process_alignment) where the
            first sequence in the alignment is the template sequence. Otherwise,
            this is a string giving the parent sequence.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        all_combos: list of tuples: All combos to encode at the given positions.
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
        batch_size: int (default = 4): The number of sequences processed in 
            each batch. The default is too low for most models, but all models
            should be able to fit on a reasonable GPU at it.
        n_processing_batches: int (default = 1): To speed up calculations, the
            tokens for all mutants are precomputed; this can require a decent 
            amount of RAM for large sequences. To save RAM, calculation of tokens
            for different mutants can be split into this number of batches.
            
        Returns
        -------
        extracted_probs: 3d numpy array: Shape is (n_combos, n_target_positions, 
            self.alphabet_size). The probabilities of the combo token occuring
            at the requested position given all other sequence context. 
        extracted_reprs: 3d numpy array: Shape is (n_combos, n_target_positions, 
            self.encoding_dim). The representations extracted for each of the
            tokens in each combination.        
        """
        # Break the set of combos
        all_chunks = np.array_split(all_combos, n_processing_batches)

        # Loop over the chunks
        chunk_probs = [None] * n_processing_batches
        chunk_reprs = [None] * n_processing_batches
        for i, chunk in tqdm(enumerate(all_chunks), total = n_processing_batches,
                             leave = True, desc = "Processing Batch"):

            # Build all mutant tokens
            mutant_tokens = self._build_mutant_tokens(sequence, target_positions,
                                                      chunk, parent_combo,
                                                      full_col = False)
            
            # Encode and record the results
            chunk_probs[i], chunk_reprs[i] = self._encode_tokens(mutant_tokens, target_positions,
                                                                 batch_size = batch_size)
            
        # Concatenate and return the different chunks
        return np.concatenate(chunk_probs), np.concatenate(chunk_reprs)
        
    def zero_shot_pred(self, sequence, target_positions, all_combos, parent_combo,
                       batch_size = 4, naive = True, full_col = False):
        """
        Uses a mask-filling protocol to return a probability of each combo in a
        combinatorial library. Combinations with higher probability can be 
        assumed to be more likely to be fit, and so this function can be used
        for making zero-shot predictions.
        
        Parameters
        ----------
        sequence: str or list of lists: If using an MsaTransformer, this is a 
            processed alignment (see support_funcs.process_alignment) where the
            first sequence in the alignment is the template sequence. Otherwise,
            this is a string giving the parent sequence.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        all_combos: list of tuples: All combos for which to perform zero-shot
            prediction
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
        batch_size: int (default = 4): The number of sequences processed in 
            each batch. The default is too low for most models, but all models
            should be able to fit on a reasonable GPU at it.
        naive: bool (default = True): Whether to use conditional or naive 
            probability when performing calculations. Naive is significantly 
            faster than conditional and yields better results on the dataset we
            tested against.
        full_col: bool (default = False): Only applicable when using the MSA
            Transformer. If True, then the full column in an alignment is masked
            at all target positions. Otherwise, only the reference sequence is
            masked at target positions.        
            
        Returns
        -------
        zero_shot_preds: 1d numpy array: The mask-filled probability for the 
            combinations given in all_combos. The order of this array is the
            same as given in the all_combos dictionary.
        """
        # Check inputs
        self._check_combo_info(target_positions, all_combos, parent_combo)
        
        # Get the log probs and a dictionary relating the masked combo to the appropriate index
        log_probs, masked_combo_to_ind = self._get_masked_log_probs(sequence, target_positions, parent_combo,
                                                                    naive = naive, batch_size = batch_size,
                                                                   full_col = full_col)
        
        # Calculate the zero-shot scores
        if naive:
            return self._zero_shot_naive(log_probs, all_combos)
        else:
            return self._zero_shot_conditional(log_probs, all_combos,
                                               masked_combo_to_ind)
        
    def _check_combo_info(self, target_positions, all_combos, parent_combo):
        """
        Checks to be sure that the sizes of the parent_combo, all_combos, and
        target_positions are all compatible.
        """
        # Make checks on combo lengths
        n_positions = len(target_positions)
        assert len(parent_combo) == n_positions, "Mismatch between combo length and N targets"
        assert all(len(combo) == n_positions for combo in all_combos), "Error in combo creation"
        
    def _build_mutant_tokens(self, sequence, target_positions, all_combos,
                             parent_combo, full_col = False):
        """
        To save time and limit back-and-forth talking between the CPU and GPU
        during calculations, mutant tokens are precomputed; this function handles
        making all mutant tokens for a given set of combinations.
        
        Parameters
        ----------
        sequence: str or list of lists: If using an MsaTransformer, this is a 
            processed alignment (see support_funcs.process_alignment) where the
            first sequence in the alignment is the template sequence. Otherwise,
            this is a string giving the parent sequence.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        all_combos: list of tuples: Each combo for which to build mutant tokens
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
        full_col: bool (default = False): Only applicable when using the MSA
            Transformer. If True, then the full column in an alignment is masked
            at all target positions. Otherwise, only the reference sequence is
            masked at target positions.        
        
        Returns
        -------
        all_tokens: 2d or 3d pytorch tensor. The tokenized mutant sequences 
            ready to be passed through `self.model`. This should be 3d if coming
            from the MsaTransformer; otherwise it is 2d.
        """  
        # Check inputs
        self._check_combo_info(target_positions, all_combos, parent_combo)
        
        # Define counts
        n_combos = len(all_combos)
        
        # Get the base tokenization and then check it
        base_tokenization = self._build_base_tokenization(sequence)
        self._check_base_tokenization(base_tokenization, sequence, target_positions, parent_combo)

        # Copy the base tokenization to include all possible combos
        repeat_args = [1] * self.token_dim
        repeat_args[0] = n_combos
        all_tokens = base_tokenization.repeat(*repeat_args)

        # Now we loop over the ind to combo dictionary and update the tokens
        # to match the target mutations
        for combo_ind, combo in enumerate(all_combos):

            # Loop over the target positions and change the value of the token
            for mutation_pos, mutation_char, parent_char in zip(target_positions, combo, parent_combo):

                # GENERIC METHOD FOR UPDATING A TOKEN TENSOR. THIS CHANGES BASED ON THE DOWNSTREAM 
                # IMPLEMENTATION
                self._make_token_mutation(all_tokens, combo_ind, mutation_pos, parent_char, 
                                          mutation_char, full_col = full_col)
                
        # Make sure we have all unique tokenizations
        assert len(torch.unique(all_tokens, dim = 0)) == n_combos, "Non-unique combos found"
        
        return all_tokens
    
    def _encode_tokens(self, all_tokens, target_positions, batch_size = 4, 
                       pbar_pos = 1):
        """
        Extracts the representations and probabilities for a set of tokens.
        
        Parameters
        ----------
        all_tokens: 2d or 3d pytorch tensor. The tokenized mutant sequences 
            ready to be passed through `self.model`. This should be 3d if coming
            from the MsaTransformer; otherwise it is 2d.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        batch_size: int (default = 4): The number of sequences processed in 
            each batch. The default is too low for most models, but all models
            should be able to fit on a reasonable GPU at it.
        pbar_pos: int (default = 1): Where to put the progress bar on stdout.
        
        Returns
        -------
        extracted_probs: 3d numpy array: Shape is (n_combos, n_target_positions, 
            self.alphabet_size). The probabilities of the combo token occuring
            at the requested position given all other sequence context. 
        extracted_reprs: 3d numpy array: Shape is (n_combos, n_target_positions, 
            self.encoding_dim). The representations extracted for each of the
            tokens in each combination.  
        """
        # Define general variables
        n_seqs = len(all_tokens)
        n_target_positions = len(target_positions)
        target_positions = torch.from_numpy(target_positions).to(self.device)
        pbar = tqdm(desc = "Sequences handled", total = n_seqs, 
                    position = pbar_pos, leave = True)

        # Define the output arrays
        extracted_probs = np.empty([n_seqs, n_target_positions, self.alphabet_size])
        extracted_reprs = np.empty([n_seqs, n_target_positions, self.encoding_dim])

        # Create a dataset and dataloader for our tokens
        pin_memory = True if self.device == "cuda" else False
        dset = TensorDataset(all_tokens)
        dloader = DataLoader(dset, batch_size = batch_size, shuffle = False,
                            pin_memory = pin_memory, drop_last = False)

        # Loop over the dataset
        sequences_handled = 0
        for (batch_tokens,) in dloader:

            # Get the batch size
            n_in_batch = len(batch_tokens)

            # Send to the appropriate device
            batch_tokens = batch_tokens.to(self.device)

            # Get the logits and representations for the batch.
            # IMPLEMENTATION VARIES
            all_temp_logits, all_temp_reprs = self._get_logits_reprs(batch_tokens, n_in_batch)
            
            # Convert logits to probabilities (while still on the GPU)
            temp_extracted_probs = F.softmax(all_temp_logits[:, target_positions], dim = -1)
            temp_extracted_reprs = all_temp_reprs[:, target_positions]
            
            # We expect the outputs to be 3D tensors
            assert temp_extracted_probs.shape == (n_in_batch, n_target_positions, self.alphabet_size)
            assert temp_extracted_reprs.shape == (n_in_batch, n_target_positions, self.encoding_dim)
            
            # Send results to numpy CPU
            upper_limit = sequences_handled + n_in_batch
            extracted_probs[sequences_handled: upper_limit] = temp_extracted_probs.cpu().numpy()
            extracted_reprs[sequences_handled: upper_limit] = temp_extracted_reprs.cpu().numpy()

            # Update the number of sequences handled
            pbar.update(n_in_batch)
            sequences_handled += n_in_batch
            
        pbar.close()
            
        return extracted_probs, extracted_reprs
        
    def _get_masked_log_probs(self, sequence, target_positions, parent_combo, 
                              batch_size = 4, naive = True, full_col = False):
        """
        Calculates the log probabilities of all possible masked combinations 
        at a set of masked positions in a sequence. The log-probabilities can be
        calculated in a naive manner (i.e. assuming all positions are
        independent of one another) or a conditional manner (i.e. calculating
        probability with the assumption that each position influences the amino
        acids that can exist at the others). When using the MSA transformer, it
        is also possible to toggle whether masking occurs just in the reference
        sequence or along the full alignment columns corresponding to the target
        positions.
        
        Parameters
        ----------
        sequence: str or list of lists: If using an MsaTransformer, this is a 
            processed alignment (see support_funcs.process_alignment) where the
            first sequence in the alignment is the template sequence. Otherwise,
            this is a string giving the parent sequence.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
        batch_size: int (default = 4): The number of sequences processed in 
            each batch. The default is too low for most models, but all models
            should be able to fit on a reasonable GPU at it.
        naive: bool (default = True): Whether to use conditional or naive 
            probability when performing calculations. Naive is significantly 
            faster than conditional and yields better results on the dataset we
            tested against.
        full_col: bool (default = False): Only applicable when using the MSA
            Transformer. If True, then the full column in an alignment is masked
            at all target positions. Otherwise, only the reference sequence is
            masked at target positions.    
            
        Returns
        -------
        log_probs: 1d numpy array: The log probabilities of each token at each
            position in all masked combinations
        masked_combo_to_ind: dict: A dictionary linking the masked combination
            to each index in the `log_probs` array.
        """
        # Define general variables
        n_target_positions = len(target_positions)

        # First, build the set of all masked positions. This is all possible combinations
        # in the dataset that have a "mask" token in them if using conditional prob.
        # Otherwise, this is just the 4-member combo
        if naive:
            filtered_combos = (tuple([self.mask_string] * n_target_positions),)
        else:
            all_strings = list(ALL_AAS) + [self.mask_string]
            all_combos = itertools.product(all_strings, repeat = n_target_positions)
            filtered_combos = list(filter(lambda x: self.mask_string in x, all_combos))
        masked_combo_to_ind = {combo: i for i, combo in enumerate(filtered_combos)}

        # Tokenize
        masked_tokens = self._build_mutant_tokens(sequence, target_positions, 
                                                  filtered_combos, parent_combo,
                                                  full_col = full_col)

        # Get the masked logits
        probs, _ = self._encode_tokens(masked_tokens, target_positions,
                                       batch_size = batch_size, pbar_pos = 0)

        # Check the calculation 
        assert np.allclose(probs.sum(axis=-1), 1)
        assert probs.shape == (len(masked_combo_to_ind), n_target_positions, self.alphabet_size)

        # Retun the log probs
        return np.log(probs), masked_combo_to_ind
    
    def _zero_shot_naive(self, log_probs, all_combos):
        """
        Given a log probability matrix of masked combinations, uses naive 
        probability to perform a zero-shot prediction of all unmasked combinations
        given in `all_combos`.
        
        Parameters
        ----------
        log_probs: 3d numpy array: This should take the shape (1, combo_size, 
            alphabet_size). Contains the log_probs of the fully combination
            (<MASK>, <MASK>, <MASK>, <MASK>)
        all_combos: list of tuples: Contains each for which to perform zero-shot
            prediction
            
        Returns
        -------
        summed_log_probs: 1d numpy array: The log probabilities of each combo
            found in `all_combos`. Higher probability means greater confidence
            that the variant is functional.
        """
        # We expect the log_probs to have shape (1, n_targets, alphabet_size)
        assert log_probs.shape == (1, len(all_combos[0]), self.alphabet_size)
        
        # For each combo, we now need to sum up the log probs
        summed_log_probs = np.zeros(len(all_combos))
        for combo_ind, combo in enumerate(all_combos):

            # Loop over the combo and add the probability to the 
            # appropriate element of the output array
            for char_ind, char in enumerate(combo):
                tok_idx = self.tok_to_idx[char]
                summed_log_probs[combo_ind] += log_probs[0, char_ind, tok_idx]

        return summed_log_probs
    
    def _zero_shot_conditional(self, log_probs, all_combos, masked_combo_to_ind):
        """
        Given a log probability matrix of masked combinations, uses conditional 
        probability to perform a zero-shot prediction of all unmasked combinations
        given in `all_combos`.
        
        Parameters
        ----------
        log_probs: 3d numpy array: This should take the shape (n_masked_combos,
            combo_size, alphabet_size). Contains the log_probs of all possible 
            masked combinations.
        all_combos: list of tuples: Contains each for which to perform zero-shot
            prediction
        masked_combo_to_ind: dict: Dictionary giving the location of the log-prob
            array corresponding to each possible masked combination. 
            
        Returns
        -------
        summed_log_probs: 1d numpy array: The log probabilities of each combo
            found in `all_combos`. Higher probability means greater confidence
            that the variant is functional.
        """
        # Define generic variables
        combo_size = len(all_combos[0])
        n_combos = len(all_combos)

        # We expect the log_probs to have shape (n_masked_combos, n_targets, alphabet_size)
        assert log_probs.shape == (len(masked_combo_to_ind), combo_size, self.alphabet_size)

        # We first need to define all possible paths toward building a 
        # combination
        all_paths = tuple(itertools.permutations(range(combo_size)))

        # Now define an output array for storing the sum of log probs
        # along all paths. 
        summed_log_probs = np.zeros([n_combos, len(all_paths)])

        # Now loop over all combinations
        for combo_ind, combo in enumerate(all_combos):

            # Now loop over all paths
            for path_ind, path in enumerate(all_paths):

                # Define a starting context
                context = [self.mask_string] * combo_size

                # Now make all steps and record probability
                for query_pos in path:

                    # Make sure the current position is masked in the context
                    assert context[query_pos] == self.mask_string

                    # Get the index in the log probs
                    mask_prob_ind = masked_combo_to_ind[tuple(context)]

                    # Get the index of the query character
                    query_char = combo[query_pos]
                    query_char_ind = self.tok_to_idx[query_char]

                    # Extract and add the log prob of the query character
                    # at the query position
                    summed_log_probs[combo_ind, path_ind] += log_probs[mask_prob_ind, 
                                                                       query_pos,
                                                                       query_char_ind]

                    # Update the context
                    context[query_pos] = query_char

        # Now we need to sum the actual probabilities to combine the different
        # conditions. 
        summed_probs = np.exp(summed_log_probs)
        return np.log(np.sum(summed_probs, axis = 1))
    
    def _check_base_tokenization(self, base_tokenization, sequence, target_positions, parent_combo):
        """
        Helper function that performs a number of checks on the tokenization of
        the reference (parent) sequence. 
        
        Parameters
        ----------
        base_tokenization: 2d or 3d torch tensor. This is 3d if using the MSA
            transformer, 2d otherwise. Contains the tokenization of the parent
            sequence. 
        sequence: str or list of lists: If using an MsaTransformer, this is a 
            processed alignment (see support_funcs.process_alignment) where the
            first sequence in the alignment is the template sequence. Otherwise,
            this is a string giving the parent sequence.
        target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein sequence, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
            
        Returns
        -------
        None
        """
        # Get the sequence length
        seq_len = len(sequence)
        
        # Confirm the expected tokenization dimensionality
        tokenization_dim = len(base_tokenization.shape)
        assert base_tokenization.shape[0] == 1, "Expected 1 element in first dimension"
        assert tokenization_dim == self.token_dim, "Incorrect token dim"
        
        # Confirm that the indices point to the expected positions
        expected_tokens = torch.tensor([self.tok_to_idx[char] for char in parent_combo])
        actual_tokens = base_tokenization[0, torch.from_numpy(target_positions)]
        assert torch.equal(expected_tokens, actual_tokens), "Unaligned parent combo and mutant positions"

        # Our code assumes the addition of a cls token. check this here
        tokenized_seq_len = base_tokenization.shape[1]
        greater1_check = tokenized_seq_len == (seq_len + 1)
        greater2_check = tokenized_seq_len == (seq_len + 2)
        assert greater1_check or greater2_check, "Expect addition of cls token"
        assert base_tokenization[0, 0] == self.tok_to_idx[self.cls_string], "Expect addition of cls"

        # If the tokenized length is 2 greater, this must be a <eos> token
        if greater2_check:
            assert base_tokenization[0, -1] == self.tok_to_idx[self.eos_string], "Expect eos token"

        # Now confirm correct base tokenization
        assert all(self.tok_to_idx[char] == base_tokenization[0, i] for 
                  i, char in enumerate(sequence, 1)), "Tokenization does not represent sequence"
        
        # Record the expected shape as an instance variable
        self._tokenization_shape = (tokenized_seq_len,)
        
    def _check_raw_logits_reprs(self, all_temp_logits, all_temp_reprs, n_in_batch):
        """
        Helper function that checks to be sure the output logits and representations
        from a transformer model have the correct shapes.
        """
        # Check shapes
        assert all_temp_logits.shape == (n_in_batch, *self.tokenization_shape, self.alphabet_size)
        assert all_temp_reprs.shape == (n_in_batch, *self.tokenization_shape, self.encoding_dim)
        
    
    def _make_token_mutation(self, all_tokens, combo_ind, mutation_pos, 
                             parent_char, mutation_char, full_col = False):
        """
        Substitutes a single position in the `all_tokens` tensor. This is used
        in a loop to build the full set of mutant tokens for a combinatorial 
        library.
        
        Parameters
        ----------
        all_tokens: 2d or 3d pytorch tensor. Contains a copy of the base tokenization
            for as many sequences as are present in a combiantorial library.
            This should be 3d if coming from the MsaTransformer; otherwise it
            is 2d.
        combo_ind: int: The index of the combo for which we are building the
            mutant tokens.
        mutation_pos: int: The position in the tokenized sequence at which we are
            making a mutation.
        parent_char: str: The expected parent character at the position to be
            mutated. This is used as a sanity check to make sure we are mutating
            the correct position.
        mutation_char: str: The mutant character.
        full_col: bool (default = False): Only applicable when using the MSA
            Transformer. If True and the mutation character is a <MASK> token,
            then the full column in an alignment is masked at the mutation_pos.
            Otherwise, only the reference sequence is masked. 
            
        Returns
        -------
        None. The all_tokens array is modified in place and will be changed but
        not returned.
        """
        # Make sure that the parent character is as we expect
        assert all_tokens[combo_ind, mutation_pos] == self.tok_to_idx[parent_char]

        # Update the tokenization tensor with the appropriate token for the mutant
        all_tokens[combo_ind, mutation_pos] = self.tok_to_idx[mutation_char]
        
    ### Abstract methods ###
    @abstractmethod
    def _load_model(self, model_name = None):
        """
        Loads the pytorch model as well as a number of instance variables about 
        the model.
        """
        self.model = None
        pass
    
    @abstractmethod
    def _build_base_tokenization(self, sequence):
        """
        Tokenizes the reference sequence. 
        """
        pass
    
    @abstractmethod
    def _get_logits_reprs(self, batch_tokens, n_in_batch):
        """
        Returns the logits and the representation for a batch of tokens.
        """
        pass
        
    ### Properties ###
    @property
    def device(self):
        return self._device
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def tok_to_idx(self):
        return self._tok_to_idx
    
    @property
    def encoding_dim(self):
        return self._encoding_dim
    
    @property
    def encoding_layer(self):
        return self._encoding_layer
    
    @property
    def token_dim(self):
        return self._token_dim
    
    @property
    def alphabet_size(self):
        return self._alphabet_size
    
    @property
    def mask_string(self):
        return self._mask_string
    
    @property
    def cls_string(self):
        return self._cls_string
    
    @property
    def eos_string(self):
        return self._eos_string
    
    @property
    def tokenization_shape(self):
        return self._tokenization_shape
    
class AbstractEsm(AbstractBertEncoder):
    """
    Abstract class for the ESM models. The MSA transformer and regular transformers
    share a number of methods/attributes that are captured by this class. 
    Specific classes for the MSA transformer and standard ESM models inherit 
    from this.
    """
    # Overwrite abstract method
    def _load_model(self, model_name):
        """
        Loads the model and gathers information on the alphabet for the model.
        Also sets a number of instance variables. Note that this function is run
        during initialization of the AbstractBertEncoder.
        
        Attributes set:
        self.model: Pytorch model for encoding and gathering logits
        self.batch_converter: Function that converts sequences into tokens
        
        Parameters
        ----------
        model_name: str: The name of the model to load. Should be available in 
            the facebookresearch/esm torchhub repo.
            
        Returns
        -------
        tok_to_idx: dict: A dicitonary linking a character to its tokenized index
        mask_string: str: The string value for the masking token used by this model
        cls_string: str: The string value for the cls token used by this model
        eos_string: str: The string value for the eos token used by this model
        """
        # Load in the model and get the batch converter
        self.model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
        self.batch_converter = alphabet.get_batch_converter()
        
        # Get the mask string
        mask_string = alphabet.get_tok(alphabet.mask_idx)
        cls_string = alphabet.get_tok(alphabet.cls_idx)
        eos_string = alphabet.get_tok(alphabet.eos_idx)
        
        # Return the model info
        return alphabet.tok_to_idx.copy(), mask_string, cls_string, eos_string
        
    def _get_logits_reprs(self, batch_tokens, n_in_batch):
        """
        Returns the logits and the representation for a batch of tokenized
        sequences.
        
        Parameters
        ----------
        batch_tokens: pytorch tensor: Will be either a 3d or 4d tensor with shape
            (batch_size, seq_len, alphabet_size) for the non-MSA transformer and shape
            (batch_size, alignment_len, seq_len, alphabet_size) for the MSA
            transformer
        n_in_batch: int: The number of sequences present in the current batch.
        
        Returns
        -------
        all_temp_logits: torch tensor: The logits for all tokens input in the
            batch. This has the same shape as `batch_tokens`
        all_temp_reprs: torch tensor: The representations for all tokens input in the
            batch. This has the same shape as `batch_tokens`, only the last 
            dimension is the representations for the token.
        """
        # Process the tokens with gradient calcs off
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers = [self.encoding_layer])

        # Confirm that the results have the expected shape
        all_temp_logits = results["logits"]
        all_temp_reprs = results["representations"][self.encoding_layer]
        
        # Check shapes
        self._check_raw_logits_reprs(all_temp_logits, all_temp_reprs, n_in_batch)
        
        return all_temp_logits, all_temp_reprs
    
################################################################################
################################## Classes #####################################
################################################################################

class EsmEncoder(AbstractEsm):
    """
    This is the class that handles calculations for the non-MSA transformer
    models in the ESM repo. It inherits most of its functionality.
    """   
    # Overwrite abstract method
    def _build_base_tokenization(self, sequence):
        """
        Tokenizes the input reference sequence.
        
        Parameters
        ----------
        sequence: str: The string representation of the parent sequence
        
        Returns
        -------
        base_tokenization: 2d torch tensor: Will have shape (1, sequence_length).
            The tokenized representation of the input sequence.
        """
        # Tokenize the sequence
        _, _, base_tokenization = self.batch_converter([("Base", sequence)])
        
        # Return the base tokenization
        return base_tokenization
    
class EsmMsaEncoder(AbstractEsm):
    """
    This class handles all calculations for the ESM MSA transformer. Note that
    a few non-abstract inherited methods are overwritten, including 
    `_check_base_tokenization`, `_make_token_mutation`, and `_get_logits_reprs`
    """ 
    # Overwrite abstract method
    def _build_base_tokenization(self, processed_alignment):
        """
        Tokenizes the input processed alignment.
        
        Parameters
        ----------
        processed_alignment: str: The string representation of the alignment.
            The reference sequence should be the first element of the alignment.
            The alignment should have been processed by 
            `.support_funcs.process_alignment`
        
        Returns
        -------
        base_tokenization: 3d torch tensor: Will have shape (1, alignment_length,
            sequence_length). The tokenized representation of the input alignment.
        """
        # Build the base tokenization, then move on to checking its integrity
        _, _, base_tokenization = self.batch_converter([processed_alignment])
        
        return base_tokenization
        
    # Overwrite _check_base_tokenization of the inherited class
    def _check_base_tokenization(self, base_tokenization, processed_alignment, msa_target_positions, parent_combo):
        """
        Helper function that performs a number of checks on the tokenization of
        the reference (parent) alignment. 
        
        Parameters
        ----------
        base_tokenization: 3d torch tensor. TContains the tokenization of the
            parent sequence output by self._build_base_tokenization.
        sequence: list of lists: This is a  processed alignment (see
            `support_funcs.process_alignment`) where the first sequence in the
            alignment is the template sequence. 
        msa_target_positions: 1d numpy array: Positions to target in the protein.
            These are 1-indexed relative to the protein alignment, 0-indexed 
            relative to the tokenized sequence (because a <cls> token is added
            to all tokenized sequences in these models)
        parent_combo: listlike: Gives the expected characters in the parent
            sequence
            
        Returns
        -------
        None
        """
        # Get the expected number of alignments and the expected sequence length
        expected_n_alignments = len(processed_alignment)
        expected_seq_len = len(processed_alignment[0][1])

        # Confirm the expected tokenization dimensionality
        tokenization_dim = len(base_tokenization.shape)
        assert base_tokenization.shape[0] == 1, "Expected 1 element in first dimension"
        assert tokenization_dim == self.token_dim, "Incorrect token dim"

        # Confirm that the indices point to the expected positions
        expected_tokens = torch.tensor([self.tok_to_idx[char] for char in parent_combo])
        actual_tokens = base_tokenization[0, 0, torch.from_numpy(msa_target_positions)] 
        assert torch.equal(expected_tokens, actual_tokens), "Unaligned parent combo and mutant positions"

        # Make sure we cover all alignments
        tokenized_n_alignments = base_tokenization.shape[1]
        assert tokenized_n_alignments == expected_n_alignments, "Incorrect tokenization of alignments"

        # Our code assumes the addition of a cls token. check this here
        tokenized_seq_len = base_tokenization.shape[2]
        greater1_check = tokenized_seq_len == (expected_seq_len + 1)
        greater2_check = tokenized_seq_len == (expected_seq_len + 2)
        assert greater1_check or greater2_check, "Expect addition of cls. Refseq length off."
        assert torch.all(base_tokenization[0, :, 0] == self.tok_to_idx["<cls>"]), "Expect addition of cls"

        # If the tokenized length is 2 greater, this must be a <eos> token
        if greater2_check:
            assert torch.all(base_tokenization[0, :, -1] == self.tok_to_idx["<eos>"]), "Expect eos token"

        # Now confirm correct base tokenization
        assert all(self.tok_to_idx[char] == base_tokenization[0, i, j]
                   for i, (_, sequence) in enumerate(processed_alignment)
                   for j, char in enumerate(sequence, 1)), "Tokenization does not represent alignment"

        # Record the expected shape as an instance variable
        self._tokenization_shape = (tokenized_n_alignments, tokenized_seq_len)

        return base_tokenization
    
    # Overwrites inherited method
    def _make_token_mutation(self, all_tokens, combo_ind, mutation_pos, 
                             parent_char, mutation_char, full_col = False):
        """
        Substitutes a single position in the `all_tokens` tensor. This is used
        in a loop to build the full set of mutant tokens for a combinatorial 
        library. Note that only the reference sequence is mutated, not all 
        members of the alignment. If building a masked combination library and
        `full_col` is set to `True`, then the full column in the alignment is
        masked.
        
        Parameters
        ----------
        all_tokens: 3d pytorch tensor. Contains a copy of the base tokenization
            for as many sequences as are present in a combiantorial library.
        combo_ind: int: The index of the combo for which we are building the
            mutant tokens.
        mutation_pos: int: The position in the tokenized sequence at which we are
            making a mutation.
        parent_char: str: The expected parent character at the position to be
            mutated. This is used as a sanity check to make sure we are mutating
            the correct position.
        mutation_char: str: The mutant character.
        full_col: bool (default = False):  If True and the mutation character is
            a <MASK> token, then the full column in an alignment is masked at
            the mutation_pos. Otherwise, only the reference sequence is masked. 
            
        Returns
        -------
        None. The all_tokens array is modified in place and will be changed but
        not returned.
        """
        # Make sure that the parent character is as we expect
        assert all_tokens[combo_ind, 0, mutation_pos] == self.tok_to_idx[parent_char]

        # If the full column flag is set and we are working with a mask token, then
        # mask update the full column. Otherwise, just update the reference sequence.
        if full_col and (mutation_char == self.mask_string):
            all_tokens[combo_ind, :, mutation_pos] = self.tok_to_idx[mutation_char]
        else:
            all_tokens[combo_ind, 0, mutation_pos] = self.tok_to_idx[mutation_char]
    
    def _get_logits_reprs(self, batch_tokens, n_in_batch):
        """
        Returns the logits and the representation for a batch of tokenized
        alignments. This extends the inherited method from AbstractEsm to handle
        the higher dimensionality of the MSA transformer outputs.
        
        Parameters
        ----------
        batch_tokens: pytorch tensor: 4d tensor with shape (batch_size,
            alignment_len, seq_len, alphabet_size). The tokens for which we will
            be generating logits and representations.
        n_in_batch: int: The number of sequences present in the current batch.
        
        Returns
        -------
        all_temp_logits: torch tensor: The logits for all tokens input in the
            batch. Only the representations for the reference sequence are returned,
            meaning that the output is a 3d tensor of shape (batch_size,
            seq_len, alphabet_size).
        all_temp_reprs: torch tensor: The representations for all tokens input in the
            batch. Only the representations for the reference sequence are returned,
            meaning that the output is a 3d tensor of shape (batch_size,
            seq_len, representation_size).
        """
        # Get the logits and representations for the full msa
        msa_temp_logits, msa_temp_reprs = super()._get_logits_reprs(batch_tokens, n_in_batch)
        
        # Check shapes
        self._check_raw_logits_reprs(msa_temp_logits, msa_temp_reprs, n_in_batch)
        
        # Get just the first position and return
        return msa_temp_logits[:, 0], msa_temp_reprs[:, 0]
    
class ProtBertEncoder(AbstractBertEncoder):
    """
    This class handles all embedding and zero-shot predictions using the 
    transformers from ProtBert. 
    """
    def _load_model(self, model_name):
        """
        Loads the model and gathers information on the alphabet for the model.
        Also sets a number of instance variables. Note that this function is run
        during initialization of the AbstractBertEncoder.
        
        Attributes set:
        self.model: Pytorch model for encoding and gathering logits
        self.tokenizer: Class that handles conversion of sequences into tokens
        
        Parameters
        ----------
        model_name: str: The name of the model to load.
            
        Returns
        -------
        tok_to_idx: dict: A dicitonary linking a character to its tokenized index
        mask_string: str: The string value for the masking token used by this model
        cls_string: str: The string value for the cls token used by this model
        eos_string: str: The string value for the eos token used by this model
        """
        # We need to load the tokenizer, the masked lm version of the model, and the 
        # encoding version of the model
        model_source = f'Rostlab/{model_name}'
        self.tokenizer = BertTokenizer.from_pretrained(model_source, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(model_source)

        # Get the token to index dictionary and the various strings
        tok_to_idx = self.tokenizer.vocab.copy()
        mask_string = self.tokenizer.mask_token
        cls_string = self.tokenizer.cls_token
        eos_string = self.tokenizer.sep_token

        return tok_to_idx, mask_string, cls_string, eos_string
    
    def _build_base_tokenization(self, sequence):
        """
        Tokenizes the input reference sequence.
        
        Parameters
        ----------
        sequence: str: The string representation of the parent sequence
        
        Returns
        -------
        base_tokenization: 2d torch tensor: Will have shape (1, sequence_length).
            The tokenized representation of the input sequence.
        """
        # First add spaces between all sequence characters
        spaced_seq = " ".join(sequence)
        assert len(spaced_seq) == (2 * len(sequence) - 1)

        # Now build the base encoding
        ids = self.tokenizer.batch_encode_plus([spaced_seq], add_special_tokens = True)

        # Now get the base tokenization
        return torch.tensor(ids["input_ids"])
            
    def _get_logits_reprs(self, batch_tokens, n_in_batch):
        """
        Returns the logits and the representation for a batch of tokenized
        sequences.
        
        Parameters
        ----------
        batch_tokens: pytorch tensor: Will be a 3d tensor with shape
            (batch_size, seq_len, alphabet_size)
        n_in_batch: int: The number of sequences present in the current batch.
        
        Returns
        -------
        all_temp_logits: torch tensor: The logits for all tokens input in the
            batch. This has the same shape as `batch_tokens`
        all_temp_reprs: torch tensor: The representations for all tokens input in the
            batch. This has the same shape as `batch_tokens`, only the last 
            dimension is the representations for the token.
        """
        # Run through the model
        with torch.no_grad():
            results = self.model(input_ids = batch_tokens, output_hidden_states = True)
            
        # Confirm that the results have the expected shape
        all_temp_logits = results.logits
        all_temp_reprs = results.hidden_states[-1]
        
        # Check shapes
        self._check_raw_logits_reprs(all_temp_logits, all_temp_reprs, n_in_batch)
            
        # Return the appropriate results
        return all_temp_logits, all_temp_reprs
    
# Make a dictionary that distributes the appropriate class to the appropriate
# model
TRANSFORMER_TO_CLASS = {"esm1b_t33_650M_UR50S": EsmEncoder,
                        "esm1_t34_670M_UR50S": EsmEncoder,
                        "esm1_t34_670M_UR50D": EsmEncoder,
                        "esm1_t34_670M_UR100": EsmEncoder,
                        "esm1_t12_85M_UR50S": EsmEncoder,
                        "esm1_t6_43M_UR50S": EsmEncoder,
                        "esm_msa1_t12_100M_UR50S": EsmMsaEncoder,
                        "prot_bert_bfd": ProtBertEncoder,
                        "prot_bert": ProtBertEncoder}