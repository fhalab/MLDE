#!/bin/bash
# This script will generate all encodings
declare -a unlearned=("onehot" "georgiev")
declare -a tape=("resnet" "bepler" "unirep" "transformer" "lstm")
declare -a others=("esm1b_t33_650M_UR50S" "prot_bert_bfd")

# Activate the conda mlde environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlde

# Define common arguments
fasta_loc="./code/validation/basic_test_data/2GI9.fasta"
alignment_loc="./code/validation/basic_test_data/GB1_Alignment.a2m"
evmut_mod_loc="./code/validation/basic_test_data/GB1_EVcouplingsModel.model"
output_loc="/mnt/Data/BJW/MLDE/RepeatEmbeddings2_ZeroShots"

# Define a function for running a learned embedding
run_learned () {
    python generate_encoding.py "$1" GB1_T2Q --fasta "$fasta_loc" \
    --positions V39 D40 G41 V54 --output "$output_loc" --batch_size 16
}

# Run generate_encoding.py for the unlearned encodings
for encoding in "${unlearned[@]}"; do
    echo "$encoding"
    python generate_encoding.py "$encoding" GB1_T2Q --n_combined 4 \
    --output "$output_loc"

    sleep 2
done

# Run generate_encoding.py for the TAPE embeddings
for encoding in "${tape[@]}"; do
    echo "$encoding"
    run_learned "$encoding"
done

# Activate mlde2
conda deactivate
conda activate mlde2

# Run generate_encoding.py for the other models
for encoding in "${others[@]}"; do
    echo "$encoding"
    run_learned "$encoding"
done

python generate_encoding.py esm_msa1_t12_100M_UR50S GB1_T2Q \
--fasta "$alignment_loc" --positions V39 D40 G41 V54 --output "$output_loc" --batch_size 8

# Run zero-shot predictions with non-deep sequence models
python predict_zero_shot.py --positions V39 D40 G41 V54 \
--models EVmutation esm_msa1_t12_100M_UR50S esm1b_t33_650M_UR50S esm1_t34_670M_UR50S \
esm1_t34_670M_UR50D esm1_t34_670M_UR100 esm1_t12_85M_UR50S esm1_t6_43M_UR50S prot_bert_bfd prot_bert \
--fasta "$fasta_loc" --alignment "$alignment_loc" --evmutation_model "$evmut_mod_loc" \
--include_conditional --output "$output_loc" --batch_size 8

# Run zero-shot predictions with DeepSequence
conda deactivate
conda activate deep_sequence
python run_deepsequence.py "$alignment_loc" --positions V39 D40 G41 V54 \
--output "$output_loc" --no_cudnn
