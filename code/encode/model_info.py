# Import required modules
import os

# Get the directory of this file
_filedir = os.path.dirname(os.path.abspath(__file__))

# Define the base weight location
_base_weight_loc = os.path.join(_filedir, "..", "tape-neurips2019", "pretrained_models")

# Define the location of pretrained weights for each model in the TAPE package
TAPE_MODEL_LOCATIONS = {"resnet": os.path.join(_base_weight_loc, "resnet_weights.h5"),
                        "bepler": os.path.join(_base_weight_loc, "bepler_unsupervised_pretrain_weights.h5"),
                        "unirep": os.path.join(_base_weight_loc, "unirep_weights.h5"),
                        "transformer": os.path.join(_base_weight_loc, "transformer_weights.h5"),
                        "lstm": os.path.join(_base_weight_loc, "lstm_weights.h5")}

# Define information pertaining to the transformers from ESM and ProtBert
TRANSFORMER_INFO = {"esm1b_t33_650M_UR50S": (1280, 33, 2),
                    "esm1_t34_670M_UR50S": (1280, 34, 2),
                    "esm1_t34_670M_UR50D": (1280, 34, 2),
                    "esm1_t34_670M_UR100": (1280, 34, 2),
                    "esm1_t12_85M_UR50S": (768, 12, 2),
                    "esm1_t6_43M_UR50S": (768, 6, 2),
                    "esm_msa1_t12_100M_UR50S": (768, 12, 3),
                    "prot_bert_bfd": (1024, None, 2),
                    "prot_bert": (1024, None, 2)}

# Define the number of latent dimensions in all models
N_LATENT_DIMS = {model: info[0] for model, info in TRANSFORMER_INFO.items()}
N_LATENT_DIMS.update({"resnet": 256, 
                      "bepler": 100, 
                      "unirep": 1900,
                      "transformer": 512,
                      "lstm": 2048})