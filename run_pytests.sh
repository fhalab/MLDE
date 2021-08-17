#!/bin/bash
# This bash script will run all pytest tests within the different environments

# Define the test files that will be tested with the mlde environment, the mlde2
# environment, and both environments
declare -a both_env_tests=(
    ./code/validation/pytest/encode/test_encoding_generator.py
    ./code/validation/pytest/encode/test_sequence_loader.py
    ./code/validation/pytest/encode/test_support_funcs.py
)

declare -a mlde2_only_tests=(
    ./code/validation/pytest/encode/test_transformer_classes.py
    ./code/validation/pytest/zero_shot
)

declare -a mlde_only_tests=(
    ./code/validation/pytest/run
)

# Write a function that evaluates a set of tests
execute_tests () {
    input_array=("$@")
    for test_loc in "${input_array[@]}"; do
        pytest "$test_loc"
    done
}

# Activate the conda mlde environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlde

# Run pytest on the appropriate mlde only and both environments tests
execute_tests "${both_env_tests[@]}"
execute_tests "${mlde_only_tests[@]}"

# Activate the mlde2 environment
conda deactivate
conda activate mlde2

# Run pytest on the appropriate mlde2 only and both environments tests
execute_tests "${both_env_tests[@]}"
execute_tests "${mlde2_only_tests[@]}"

# Activate the deepsequence environment
conda deactivate
conda activate deep_sequence

# Run deep sequence tests
pytest deep_sequence/test_deepseq.py