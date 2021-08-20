MLDE
====
A machine-learning package for navigating combinatorial protein fitness landscapes. This repository accompanies our work "[Informed training set design enables efficient machine learning-assisted directed protein evolution](https://doi.org/10.1016/j.cels.2021.07.008)".

Table of Contents
-----------------
- [MLDE](#mlde)
  - [Table of Contents](#table-of-contents)
- [MLDE Concept](#mlde-concept)
- [V1.0.0 What's New](#v100-whats-new)
  - [Major Changes](#major-changes)
  - [Minor Changes](#minor-changes)
  - [Known Limitations](#known-limitations)
- [Installation](#installation)
  - [Installation Validation](#installation-validation)
    - [Basic Tests](#basic-tests)
    - [Pytest Validation](#pytest-validation)
- [General Use](#general-use)
  - [Generating Encodings with generate_encoding.py](#generating-encodings-with-generate_encodingpy)
    - [Inputs for generate_encoding.py](#inputs-for-generate_encodingpy)
    - [Examples for generate_encoding.py](#examples-for-generate_encodingpy)
    - [Outputs for generate_encoding.py](#outputs-for-generate_encodingpy)
  - [Zero Shot Prediction with predict_zero_shot.py and run_deepsequence.py](#zero-shot-prediction-with-predict_zero_shotpy-and-run_deepsequencepy)
    - [Inputs for predict_zero_shot.py](#inputs-for-predict_zero_shotpy)
    - [Inputs for run_deepsequence.py](#inputs-for-run_deepsequencepy)
    - [Building an Alignment for DeepSequence and Obtaining an EVmutation Model](#building-an-alignment-for-deepsequence-and-obtaining-an-evmutation-model)
    - [Building an Alignment for MSA Transformer](#building-an-alignment-for-msa-transformer)
    - [Examples for predict_zero_shot.py](#examples-for-predict_zero_shotpy)
    - [Examples for run_deep_sequence.py](#examples-for-run_deep_sequencepy)
    - [Outputs for predict_zero_shot.py and run_deep_sequence.py](#outputs-for-predict_zero_shotpy-and-run_deep_sequencepy)
  - [Making Predictions with execute_mlde.py](#making-predictions-with-execute_mldepy)
    - [Inputs for execute_mlde.py](#inputs-for-execute_mldepy)
      - [TrainingData.csv](#trainingdatacsv)
      - [Custom Encodings](#custom-encodings)
      - [MldeParameters.csv](#mldeparameterscsv)
    - [Examples for execute_mlde.py](#examples-for-execute_mldepy)
    - [Outputs for execute_mlde.py](#outputs-for-execute_mldepy)
  - [Replicating Published Results with simulate_mlde.py](#replicating-published-results-with-simulate_mldepy)
- [Program Details](#program-details)
  - [Inbuilt Models](#inbuilt-models)
- [Dependencies](#dependencies)
  - [OS](#os)
  - [Hardware](#hardware)
  - [Software](#software)
    - [MLDE Software](#mlde-software)
    - [DeepSequence](#deepsequence)
- [Citing this Repository](#citing-this-repository)
- [Citing Supporting Repositories](#citing-supporting-repositories)

# MLDE Concept
MLDE attempts to learn a function that maps protein sequence to protein fitness for a combinatorial design space. The procedure begins by experimentally measuring the fitness values of a small subsample from a combinatorial library (e.g., a library built using “NNK” degenerate primers to make mutations at multiple positions simultaneously). These "labeled" variants are then used to train an ensemble of 22 models with varied architectures (see [Inbuilt Models](#inbuilt-models)). Trained models are next ranked by validation error, and the predictions of the N-best models are averaged to predict fitness values for the unsampled (“unlabeled”) variants. Unlabeled variants are ranked according to predicted fitness and returned. More detailed information on programmatic implementation is given [below](#program-details).

# V1.0.0 What's New

## Major Changes
1. Higher capacity, more diverse models for encoding: In addition to the models made available in V0.0.0 (those from the [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019#pretrained-models) repository), V1.0.0 can also use all models from [Facebook Research ESM V0.3.0](https://github.com/facebookresearch/esm/tree/v0.3.0#pre-trained-models-) and the ProtBert-BFD and ProtBert models from [ProtTrans](https://github.com/agemagician/ProtTrans#%EF%B8%8F-models-availability).
2. Zero-shot prediction: We have added script support for zero-shot prediction using various sequence-based strategies. We have found that eliminating holes from training data greatly improves MLDE outcome. This focused training MLDE (ftMLDE) can be accomplished by using zero-shot prediction strategies to focus laboratory screening efforts on variants with higher probability of retaining function. From simple command line inputs, MLDE V1.0.0 enables zero-shot prediction of the fitness of all members of combinatorial design spaces using [EVmutation](https://doi.org/10.1038/nbt.3769), [DeepSequence](https://doi.org/10.1038/s41592-018-0138-4), and masked token filling using models from [ESM](https://github.com/facebookresearch/esm/tree/v0.3.0#evolutionary-scale-modeling) and [ProtTrans](https://github.com/agemagician/ProtTrans#prottrans). Details on EVmutation and DeepSequence can be found in their original papers. Details on mask filling can be found in our accompanying [paper](). We welcome additional suggestions on zero-shot strategies to wrap in the MLDE pipeline.
3. To accomodate the new packages and code, two new conda environment files have been provided (mlde2.yml and deep_sequence.yml).
4. Addition of a script for replicating simulations performed in our accompanying paper.

## Minor Changes
1. To fix an occasional dying ReLU problem, all activations in the neural network models have been replaced with ELU.
2. Classes, functions, filenames, and foldernames have been changed to better align with PEP8 standards.
3. The installation procedure has been simplified.

## Known Limitations
We have not tested running multiple instances of MLDE in parallel on the same machine. While running in parallel should be expected to work in most cases, there are potential conflicts surrounding DeepSequence, where a temporary file is saved with a fixed name in order to make zero-shot predictions -- processes running in parallel have the potential to overwrite the file of the other, thus resulting in incorrect results. If support for parallel runs is requested, this is something we can look into more.

# Installation
1. MLDE relies on three submodules: [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019), which accompanies the work by Rao *et al.* *Evaluating Protein Transfer Learning with TAPE*, [DeepSequence](https://github.com/debbiemarkslab/DeepSequence), which accompanies the work by Riesselman *et al.* *Deep generative models of genetic variation capture the effects of mutations*, and [ESM](https://github.com/facebookresearch/esm), which accompanies the work by Rives *et al.* *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. To include submodules, the MLDE repository must be cloned with the "--recurse-submodules" flag as below. MLDE will not work correctly without these submodules present.

```bash
git clone --recurse-submodules https://github.com/fhalab/MLDE.git
```

2. The repository will be downloaded with the `mlde.yml`, `mlde2.yml`, and `deep_sequence.yml` anaconda environment templates in the top-level directory. Due to incompatibilities between the CUDA Toolkit version required by tensorflow 1.13 (used by tape-neurips2019 and the version of Keras used to perform MLDE predictions) and Pytorch >v1.5 (required by models from ESM), we could not find a single stable environment in which new code was compatible with old. Thus, **the `mlde.yml` environment should be used for all functionality from V0.0.0 (encoding with tape-neurips 2019 models as well as making predicitons with MLDE)** and **the `mlde2.yml` environment should be used for all functionality new in V1.0.0 (encoding with ESM and ProtBert models and for making zero-shot predictions)**. Note that all environments assume you have pre-installed CUDA and have a CUDA-capable GPU on your system. If you would rather work in a separate environment, dependencies for the full MLDE codebase can be found [here](#Dependencies). To build the conda environments, run

```bash
cd ./MLDE
conda env create -f mlde.yml
conda env create -f mlde2.yml
conda env create -f deep_sequence.yml
```

The environments must be created from within the MLDE repository, otherwise [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019) will not be correctly installed.

3. Finally, we need to download the model weights needed for generating learned embeddings using [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019). Navigate to the tape-neurips2019 submodule and download the model weights as below:

```bash
cd ./code/tape-neurips2019
./download_pretrained_models.sh
```

## Installation Validation
### Basic Tests
Basic functionality of MLDE can be tested by running generate_encoding.py, predict_zero_shot.py, and execute_mlde.py with provided example data. Test command line calls are given below. To run genereate_encoding.py:

```bash
conda activate mlde
python generate_encoding.py transformer GB1_T2Q --fasta
  ./code/validation/basic_test_data/2GI9.fasta --positions V39 D40 --batches 1
```

To run predict_zero_shot.py:

```bash
conda activate mlde2
python predict_zero_shot.py --positions V39 D40 --models esm1_t6_43M_UR50S
--fasta ./code/validation/basic_test_data/2GI9.fasta --include_conditional
```

To run execute_mlde.py:

```bash
conda activate mlde
python execute_mlde.py ./code/validation/basic_test_data/InputValidationData.csv
  ./code/validation/basic_test_data/GB1_T2Q_georgiev_Normalized.npy
  ./code/validation/basic_test_data/GB1_T2Q_ComboToIndex.pkl
  --model_params ./code/validation/basic_test_data/TestMldeParams.csv
  --hyperopt
```

### Pytest Validation
MLDE has been thoroughly tested using [pytest](https://docs.pytest.org/en/stable/). These tests can be repeated by executing the bash script `run_pytest.sh` from the top-level MLDE folder as below. Note that DeepSequence tests rely on data provided in the DeepSequence GitHub repository. They can be downloaded by running the `download_alignments.sh` and `download_pretrained.sh` scripts found in the DeepSequence submodule at `deep_sequnce/DeepSequence/examples/`.

```bash
PYTHONPATH=$PathToMldeRepo ./run_pytests.sh
```

Where $PathToMldeRepo should be the path to the folder in which `run_pytests.sh` is stored. Note that these tests can take a while, so it is best to run overnight.

# General Use
MLDE works in two or three stages: Encoding generation, zero-shot prediction, and prediction. The encoding generation stage takes a parent protein sequence and generates all encodings for a given combinatorial library. The zero-shot prediction stage can be used to recommend subsets of the full combinatorial library on which laboratory screening efforts should be focused. The prediction stage uses the encodings from the first stage in combination with fitness data from the laboratory-evaluated set of combinations to make predictions on the fitness of the full combinatorial space. The MLDE package can generate all encodings presented in our accompanying paper; users can also pass in their own encodings (e.g. from a custom encoder) to the prediction stage scripts.

## Generating Encodings with generate_encoding.py
Encodings for MLDE are generated with the `generate_encoding.py` script. The combinatorial space can be encoded using one-hot, Georgiev, or learned embeddings. Learned embeddings can be generated from any of the completely unsupervised pre-trained models found in [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019#pretrained-models), [Facebook Research ESM V0.3.0](https://github.com/facebookresearch/esm/tree/v0.3.0#pre-trained-models-) or the ProtBert-BFD and ProtBert models from [ProtTrans](https://github.com/agemagician/ProtTrans#%EF%B8%8F-models-availability). The mlde environment should be used for tape-neurips2019 models, onehot, and georgiev encodings, while the mlde2 environment should be used for ESM and ProtTrans models.

### Inputs for generate_encoding.py
There are a number of required and optional arguments that can be passed to generate_encoding.py, each of which is detailed in the table below

| Argument | Type | Description |
|:---------|---------------|-------------|
| encoding | Required Positional Argument | This argument sets the encoding type used. Choices include "onehot", "georgiev", "resnet", "bepler", "unirep", "transformer", "lstm", "esm_msa1_t12_100M_UR50S", "esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S", "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100", "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S", "prot_bert_bfd", or "prot_bert". Models should be typed exactly as given in this table (including casing) when passed into generate_encoding.py.|
| protein_name | Required Positional Argument | Nickname for the protein. Will be used to prefix output files.|
| --fasta | Required Keyword Argument for Learned Embeddings | The parent protein amino acid sequence in fasta file format. If using the MSA Transformer from ESM (esm_msa1_t12_100M_UR50S), then this should be a .a2m or .a3m file where the first sequence in the alignment is the reference sequence. See the section on [Building an Alignment for MSA Transformer](#building-an-alignment-for-msa-transformer) for instructions on preparing an msa for an input.|
| --positions | Required Keyword Argument for Learned Embeddings | The positons and parent amino acids to include in the combinatorial library. Input format must be "AA# AA#" and should be in numerical order of the positions. For instance, to mutate positions Q3 and F97 in combination, the flag would be written as `--positions Q3 F97`.|
| --n_combined | Required Keyword Argument for Georgiev or Onehot Encodings | The number of positions in the combinatorial space. |
| --output | Optional Keyword Argument | Output location for saving data. Default is the current working directory if not specified. |
| --batches | Optional Keyword Argument | Generating the largest embedding spaces can require high levels of system RAM. This parameter dictates how many batches to split a job into. If not specified, the program will attempt to automatically determine the appropriate number of batches given the available RAM on your system. |
| --batch_size | Optional Keyword Argument | Sets the batch size of calculation for the ESM and ProtTrans models. If processing on a GPU, increasing batch size can result in shorter processing time. The default is "4" when not explicitly set. This default is too small for most models, and is set to accommodate the largest models on standard commercial GPUs. |

### Examples for generate_encoding.py
The below example is a command line call for generating Georgiev encodings for a 4-site combinatorial space (160000 total variants). This command could also be used for generating onehot encodings. Note that, because georgiev and onehot encodings are context-independent, no fasta file is needed. Indeed, the encodings generated from the below call could be applied to any 4-site combinatorial space.

```bash
conda activate mlde
python generate_encoding.py georgiev example_protein --n_combined 4
```

The below example is a command line call for generating transformer embeddings for the 4-site GB1 combinatorial library discussed in our work. Note that this command could be used for generating any of the learned embeddings (with appropriate substitution of the `encoding` argument). Because embeddings are context-aware, these encodings should not be used for another protein.

```bash
conda activate mlde
python generate_encoding.py transformer GB1_T2Q
  --fasta ./code/validation/basic_test_data//2GI9.fasta
  --positions V39 D40 G41 V54 --batches 4
```

Note that the `mlde2` environment would be used for ESM and ProtTrans-based models in the above example. The input fasta file looks as below:

```
>GB1_T2Q
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```

Note that if using the MSA transformer, then the fasta file should be replaced with a .a2m file. The first few rows of this file will look something like the below (using GB1 as an example):

```
>TARGET/1-56
mqykLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKtftvte
>UniRef100_Q53975/224-278
.tykLVVKGNTFSGETTTKAIDTATAEKEFKQYATANNVDGEWSYDDATKtftvte
>UniRef100_Q53975/294-348
.tykLIVKGNTFSGETTTKAVDAETAEKAFKQYATANNVDGEWSYDDATKtftvte
>UniRef100_Q53975/364-418
.tykLIVKGNTFSGETTTKAIDAATAEKEFKQYATANGVDGEWSYDDATKtftvte
>UniRef100_Q53975/434-488
.tykLIVKGNTFSGETTTKAVDAETAEKAFKQYANENGVYGEWSYDDATKtftvte
>UniRef100_Q53975/504-558
.tykLVINGKTLKGETTTKAVDAETAEKAFKQYANENGVDGVWTYDDATKtftvte
```

### Outputs for generate_encoding.py
Every run of generate_encoding.py produces a time-stamped folder containing all results. The time-stamp format is "YYYYMMDD-HHMMSS" (Y = year, M = month, D = day, H = 24-hour, M = minute, S = second). The time-stamped folder contains subfolders "Encodings" and "Fastas".

The "Encodings" folder will contain the below files ("\$NAME" is from the `name` argument of generate_encoding.py; "\$ENCODING" is from the `encoding` argument):

| Filename | Description |
|:---------|-------------|
|\$NAME_\$ENCODING_Normalized.npy| Numpy array containing the mean-centered, unit-scaled amino acid embeddings. These are the embeddings that will typically be used for generating predictions, and take the shape $20^C x C x L$, where $C$ is the number of amino acid positions combined and $L$ is the number of latent dimensions per amino acid for the encoding.|
|\$NAME_\$ENCODING_UnNormalized.npy| Numpy array containing the unnormalized amino acid embeddings. This tensor will take the same shape as \$NAME_\$ENCODING_Normalized.npy.|
|\$NAME_\$ENCODING_ComboToIndex.pkl| A pickle file containing a dictionary linking amino acid combination to the index of that combination in the output encoding tensors. Note that combinations are reported in order of amino acid index (e.g. a combination of A14, C23, Y60, and W91 would be written as "ACYW").|
|\$NAME_\$ENCODING_IndexToCombo.pkl| A pickle file containing a dictionary that relates index in the encoding tensor to the combination.|

Note that when encoding is "onehot", only unnormalized embeddings will be returned.

The "Fastas" directory is only populated when models from TAPE are run. It contains fasta files with all sequences used to generate embeddings, split into batches as appropriate.

## Zero Shot Prediction with predict_zero_shot.py and run_deepsequence.py
We have found that eliminating holes from training data greatly improves MLDE outcome. This focused training MLDE (ftMLDE) can be accomplished by using zero-shot prediction strategies to focus laboratory screening efforts on variants with higher probability of retaining function. In our recent work, we evaluated a number of zero-shot prediction strategies, including [EVmutation](https://doi.org/10.1038/nbt.3769), [DeepSequence](https://doi.org/10.1038/s41592-018-0138-4), and mask filling using BERT-style models. We encourage users to read the original EVmutation and DeepSequence papers for details on their functionality; we detail mask filling in our [accompanying work](). In our accompanying work, we also tested ΔΔG predictions using the [Triad protein modeling software](https://www.protabit.com/#triad); this software is not open-source, however, and so we do not include it here.

### Inputs for predict_zero_shot.py
predict_zero_shot.py can be used to make zero shot predictions using a number of different models and strategies in a single command line call. This script should be run within the mlde2 environment. All arguments are given below:


| Argument | Type | Description |
|:---------|---------------|-------------|
| --positions | Required Keyword Argument | The positons and parent amino acids to include in the combinatorial library. Input format must be "AA# AA#" and should be in numerical order of the positions. For instance, to mutate positions Q3 and F97 in combination, the flag would be written as `--positions Q3 F97`.|
| --models | Required Keyword Argument | The models to use for zero-shot prediction. Options include "EVmutation", "esm_msa1_t12_100M_UR50S", "esm1b_t33_650M_UR50S", "esm1_t34_670M_UR50S", "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100", "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S", "prot_bert_bfd", or "prot_bert". Models should be typed exactly as given in this table (including casing) when passed into predict_zero_shot.py. Note that multiple models can be passed in in a single run by separating each model name with a space (e.g. `--models esm1_t34_670M_UR50S prot_bert` would provide zero-shot predictions using mask filling with both esm1_t34_670M_UR50S and protbert). |
| --fasta | Required Keyword Argument for all but the MSA transformer | The parent protein amino acid sequence in fasta file format. |
| --alignment | Required Keyword Argument for the MSA Transformer | A .a2m or .a3m file containing the parent sequence (which should be first in the alignment) and all aligned sequences. See the below section on [Building an Alignment for MSA Transformer](#building-an-alignment-for-msa-transformer) for further details|
| --evmutation_model | Required Keyword Argument for EVmutation | A model parameters file describing an EVmutation model. See the below section on [Building an Alignment for DeepSequence and Obtaining an EVmutation Model](#building-an-alignment-for-deepSequence-and-obtaining-an-evmutation-model) for further details.
| --include_conditional | Flag | By default, any mask-filling model will use naive probability for calculating zero-shot predictions. If this flag is included, then predictions using conditional probability will also be returned. |
| --mask_col | Flag | This only applies to the MSA transformer. By default, only positions in the reference sequence are masked during mask-filling. If this flag is included, then the entire alignment column is masked instead. |
| --batch_size | Optional Keyword Argument | Sets the batch size of calculation for the ESM and ProtTrans models. If processing on a GPU, increasing batch size can result in shorter processing time. The default is "4" when not explicitly set. This default is too small for most models, and is set to accommodate the largest models on standard commercial GPUs. |
| --output | Optional Keyword Argument | Output location for saving data. Default is the current working directory if not specified. |

### Inputs for run_deepsequence.py
The [DeepSequence repository](https://github.com/debbiemarkslab/DeepSequence) was written in Python 2, and so is incompatible with much of the MLDE framework. It must thus be run within its own conda environment and is executed from a stand-alone script.

Inputs to `run_deepsequence.py` are given below:


| Argument | Type | Description |
|:---------|---------------|-------------|
| alignment | Required Positional Argument | A .a2m file containing the parent sequence (which should be first in the alignment) and all aligned sequences. .a3m files will not work here. See the below section on [Building an Alignment for DeepSequence and Obtaining an EVmutation Model](#building-an-alignment-for-deepSequence-and-obtaining-an-evmutation-model) for further details.|
| --positions | Required Keyword Argument | The positons and parent amino acids to include in the combinatorial library. Input format must be "AA# AA#" and should be in numerical order of the positions. For instance, to mutate positions Q3 and F97 in combination, the flag would be written as `--positions Q3 F97`.|
| --output | Optional Keyword Argument | Output location for saving data. Default is the current working directory if not specified. |
| --save_model | Flag | Whether or not to save the model parameters of the trained VAE model. If set, then parameters will be saved to the DeepSequence submodule at /examples/params/ |
| --no_cudnn | Flag | DeepSequence runs on Theano, which is no longer a supported package. As a result, newer cuDNN libraries are no longer compatbile with its code and can cause this script to fail. If you are running into compatibility issues when running this code, set this flag to turn off use of cuDNN; this will slow down computation, but should not change the output. |

### Building an Alignment for DeepSequence and Obtaining an EVmutation Model
We recommend using the excellent [EVcouplings webserver](https://v2.evcouplings.org/) to build MSAs for DeepSequence and EVmutation. It can also be used to build MSAs for the MSA transformer, though additional processing of the output may be required to reduce the size of the output alignment to a more appropriate size for the MSA transformer (see [Building an Alignment for MSA Transformer](#building-an-alignment-for-msa-transformer) for more details).

The optimal parameters for the homology search will vary depending on the protein, and we refer users to the original [EVmutation](https://doi.org/10.1038/nbt.3769) and [DeepSequence](https://doi.org/10.1038/s41592-018-0138-4) papers for information on how to best tune them. Once the homology search is complete, alignments can be downloaded in a2m format by navigating to the "Downloads" tab of the results interface and clicking "Sequence alignment" in the "Sequence alignment" section. The model parameters for an EVmutation model trained using this alignment can also be obtained by clicking "EVcouplings model parameters"; the downloaded file should be input as the `--evmutation_model` argument in predict_zero_shot.py.

### Building an Alignment for MSA Transformer
The .a2m file output by the EVcouplings webserver can be used by predict_zero_shot.py and generate_encoding.py for zero-shot predictions and encoding generation, respectively, with the MSA transformer. As stated by the [developers of the MSA transformer](https://doi.org/10.1101/2021.02.12.430858) "At inference time...we do not provide the full MSA to the model as it would be computationally expensive and the model’s performance can decrease when the input is much larger than that used during training." Instead, the authors tested subsampling of the MSA (ensuring that the reference sequence was always included) using a variety of different strategies. We encourage users to read the [original MSA transformer paper](https://doi.org/10.1101/2021.02.12.430858) for details on recommended practices for subsampling an MSA.

### Examples for predict_zero_shot.py

predict_zero_shot.py should be run from within the mlde2 environment. The below code would make zero-shot predictions for the 4-site GB1 combinatorial library using EVmutation, the MSA Transformer, and ESM1b, considering both conditional and naive probability for the mask-filling models.

```bash
conda activate mlde2
python predict_zero_shot.py --positions V39 D40 G41 V54 --models esm1b_t33_650M_UR50S
    EVmutation esm_msa1_t12_100M_UR50S --fasta ./code/validation/basic_test_data/2GI9.fasta
    --alignment ./code/validation/basic_test_data/GB1_Alignment.a2m
    --evmutation_model ./code/validation/basic_test_data/GB1_EVcouplingsModel.model
    --include_conditional --batch_size 32 --output ~/Downloads
```

Note that not all of the flags included above need to be used; they are present just for demonstration purposes. Inclusion of `--mask_col` would result in masking the full columns when performing mask filling with the MSA transformer.

### Examples for run_deep_sequence.py

Within the deep_sequence environment, DeepSequence can be run with the below command:

```bash
conda activate deep_sequence
python run_deep_sequence.py ./code/validation/basic_test_data/GB1_Alignment.a2m
  --positions V39 D40 --output ~/Downloads --save_model --no_cudnn
```

Note that all flags/arguments are shown for sake of example. In practice, the `--output`, `--save_model`, and `--no_cudnn` flags may be ommitted and the code will still run.

### Outputs for predict_zero_shot.py and run_deep_sequence.py

Both `predict_zero_shot.py` and `run_deep_sequence.py` output a single csv file to the directory specified by `--output`. The first column in this csv file is "Combo", which contains the 4-letter shorthand notation for all combinations possible in the defined combinatorial library. For instance, the wild type combo V39, D40, G41, V54 for GB1 would be written as "VDGV" in this output. For the `run_deep_sequence.py` output, the only other column is "DeepSequence", which contains all predictions made by the trained model. For the `predict_zero_shot.py` output, all remaining columns are the outputs from the different zero-shot predictors requested with the `--models` argument along with any other relevant information about the prediction separated by a "-" delimiter. For instance, the column name "esm1_t34_670M_UR50S-Naive" would indicate that the output corresponds to predictions using the esm1_t34_670M_UR50S model from ESM using naive probability for mask filling. Outputs from EVmutation and DeepSequence are ΔELBo, while outputs from all other models are log probabilities. **For all models, a higher (less negative) zero-shot score means greater confidence that that variant will be functional.**

## Making Predictions with execute_mlde.py
MLDE predictions are made using the execute_mlde.py script. This script should be run in the mlde conda environment. Inputs to this script include the encodings for all possible members of the combinatorial space, experimentally determined sequence-function data for a small number of combinations (for use as training data), and a dictionary linking all possible members of the combinatorial space to their associated encoding.

### Inputs for execute_mlde.py
| Argument | Type | Description |
|:---------|-----------|-------------|
| training_data | Required Argument | A csv file containing the sequence-function information for sampled combinations. More information on this file can be found [below](#trainingdata.csv). |
| encoding_data | Required Argument | A numpy array containing the embedding information for the full combinatorial space. Encoding arrays generated by generate_encoding.py can be passed directly in here. Custom encodings can be passed in here too, the details of which are discussed [below](#custom-encodings). |
| combo_to_ind_dict | Required Argument | A pickle file containing a dictionary that links a combination to its index. The ComboToIndex.pkl file output by generate_encoding.py can be passed in directly here. |
| model_params | Optional Argument | A csv file dictating which inbuilt MLDE models to use as well as how many rounds of hyperparameter optimization to perform. The makeup of this file is discussed [below](#mldeparameters.csv). |
| output | Optional Argument | The location to save the results. Default is the current working directory. |
| n_averaged | Optional Argument | The number of top-performing models to average to get final prediction results. Default is 3. |
| n_cv | Optional Argument | The number of rounds of cross validation to perform during training. Default is 5. |
| no_shuffle | Flag | When set, the indices of the training data will **not** be shuffled for cross-validation. Default is to shuffle indices.|
| hyperopt | Flag | When set, hyperparameter optimization will also be performed. Note that this can greatly increase the run time of MLDE depending on the models included in the run. The default is to not perform hyperparameter optimization. |

#### TrainingData.csv
This csv file contains the sequence-function data for the protein of interest. An example csv file can be found in MLDE/code/validation/basic_test_data/InputValidationData.csv. The top few rows of this file are demonstrated below:

| AACombo | Fitness |
|:--------|---------|
| CCCC | 0.5451 |
| WCPC | 0.0111 |
| WPGC | 0.0097 |
| WGPP | 0.0022 |

The two column headers must always be present and always have the same name. Sequence is input as the combination identity, which is the amino acid present at each position in order. For instance, a combination of A14, C23, Y60, and W91 would be written as "ACYW".

While not strictly required, it is recommended to normalize fitness in some manner. Common normalization factors would be the fitness of the parent protein or the maximum fitness in the training data.

#### Custom Encodings
Any encoding can be passed into execute_mlde.py so long as it meets the dimensionality requirements. Specifically, the array must take the shape `20^C x C x L`, where `C` is the number of amino acid positions combined and `L` is the number of dimensions per amino acid for the encoding. The program will throw an exception if a non-3D encoding is passed in as `encoding_data`.

Note that for all but the convolutional neural networks, the last 2 dimensions of the input space will be flattened before processing. In other words, convolutional networks are trained on 2D encodings and all other models on 1D encodings.

#### mlde_parameters.csv
This file details what models are included in an MLDE run and how many hyperparameter optimization rounds will be executed for each model. By default, the file found at MLDE/code/params/mlde_parameters.csv is used, though users can pass in their own versions; the default file can be used as a template for custom parameter files, but should never be changed itself. The contents of the mlde_parameters.csv file are copied below:

| ModelClass | SpecificModel | Include | NHyperopt |
|:-----------|---------------|---------|-----------|
|Keras | NoHidden | TRUE | 10
|Keras | OneHidden | TRUE | 10
|Keras | TwoHidden | TRUE | 10
|Keras | OneConv | TRUE | 10
|Keras | TwoConv | TRUE | 10
|XGB | Tree | TRUE | 100
|XGB | Linear | TRUE | 100
|XGB | Tree-Tweedie | TRUE | 100
|XGB | Linear-Tweedie | TRUE | 100
|sklearn-regressor | Linear | TRUE | 100
|sklearn-regressor | GradientBoostingRegressor | TRUE | 100
|sklearn-regressor | RandomForestRegressor | TRUE | 100
|sklearn-regressor | BayesianRidge | TRUE | 100
|sklearn-regressor | LinearSVR | TRUE | 100
|sklearn-regressor | ARDRegression | TRUE | 100
|sklearn-regressor | KernelRidge | TRUE | 100
|sklearn-regressor | BaggingRegressor | TRUE | 100
|sklearn-regressor | LassoLarsCV | TRUE | 100
|sklearn-regressor | DecisionTreeRegressor | TRUE | 100
|sklearn-regressor | SGDRegressor | TRUE | 100
|sklearn-regressor | KNeighborsRegressor | TRUE | 100
|sklearn-regressor | ElasticNet | TRUE | 100

The column names should not be changed. Changing the "Include" column contents to 'FALSE' will stop a model from being included in the ensemble trained for MLDE; the same can be accomplished by simply deleting the row. The "NHyperopt" column contents can be changed to alter how many hyperparameter optimization rounds are performed when the `hyperopt` flag is thrown. Note that Keras-based models can take a long time for hyperparameter optimization, hence why only 10 rounds are performed by default.

### Examples for execute_mlde.py
The below is an example run of execute_mlde.py using information output by generate_encoding.py as its inputs. Note that all optional arguments are used here for demonstration purposes; they don't all need to be used in practice.

```bash
conda activate mlde
python execute_mlde.py .code/validation/basic_test_Data/InputValidationData.csv
    ./code/validation/basic_test_data/GB1_T2Q_georgiev_Normalized.npy
    ./code/validation/basic_test_data/GB1_T2Q_ComboToIndex.pkl
    --model_params ../code/validation/basic_test_data/TestMldeParams.csv
    --output ~/Downloads --n_averaged 5 --n_cv 10 --hyperopt
```

### Outputs for execute_mlde.py
Every run of execute_mlde.py produces a time-stamped folder containing all results. The time-stamp format is "YYYYMMDD-HHMMSS" (Y = year, M = month, D = day, H = 24-hour, M = minute, S = second). The time-stamped folder contains the files "PredictedFitness.csv", "LossSummaries.csv", "CompoundPreds.npy", "IndividualPreds.npy", and "PredictionStandardDeviation.npy". If hyperparameter optimization is performed, an additional file called "HyperoptInfo.csv" will also be generated. The contents of each file are detailed below:

| Filename | Description |
|:---------|-------------|
| PredictedFitness.csv | This csv file reports the average predicted fitness of the top models given by `n_averaged` for all possible combinations (including combinations in the training data). Whether a combination was present in the training data or not is clearly marked. |
| LossSummaries.csv | This csv file reports the cross-validation training and testing error of the best models from each class. |
| CompoundPreds.npy | This numpy file contains an array with shape `M x 20^C`, where `M` is the number of models and `C` is the number of amino acids in the combinatorial space. This array gives the average predictions of the top-M models for all possible combinations. For instance, index 0 gives the predictions of the best model; index 1 gives the average predictions of the top 2 models, and so on. |
| IndividualPreds.npy| This numpy array gives the predictions of all models, ordered by the model's cross-validation testing error. This array is the same shape as CompoundPreds.npy. |
| PredictionStandardDeviation.npy| This numpy array gives the standard deviation of predictions across the models generated from different cross-validation steps. It has the same shape and ordering as IndividualPreds.npy. |
|HyperoptInfo.csv | This csv file gives details on the hyperparameter optimization procedure, including parameter values tested and associated cross-validation errors in each iteration. |

## Replicating Published Results with simulate_mlde.py
We provide the script `simulate_mlde.py` for replicating simulations performed in [our publication](https://doi.org/10.1016/j.cels.2021.07.008). To use this script, you must first download the embeddings, training indices, cross validation indices, and other information that we used for performing simulations from CaltechData. A link to the appropriate download is [here](https://data.caltech.edu/records/1958).

The CaltechData folder contains 4 objects:
1. SimulationTrainingData
2. AllSummaries_ByModel.csv
3. AllSummaries_Ensemble.csv
4. README.txt

The csv files contain summary information for all simulations performed in our work either for predictions made by independent models (ByModel) or an ensemble of the best models (Ensemble). Relevant information about the contents of these files can be found in the README.txt file. To perform simulations using `simulate_mlde.py`, you must first move the folder "SimulationTrainingData" to the top-level MLDE directory (i.e., in the same directory as `simulate_mlde.py`).

`simulate_mlde.py` should be run in the `mlde` conda environment. Each call to `simulate_mlde.py` will perform simulations using one design condition tested in our work. For instance, a single run of `simulate_mlde.py` could be used to run the simulations using onehot encoding with 384 training points drawn from the top 6400 samples predicted by Triad. Arguments to the script are given in detail below:

| Argument | Type | Description |
|:---------|-----------|-------------|
| encoding | Required Argument | The encoding to use for running simulations. Options include "bepler", "esm1b_t33_650M_UR50S.npy", "georgiev", "lstm", "msa_transformer", "onehot", "ProtBert_BFD", "resnet", "transformer", and "unirep". |
| training_type | Required Argument | The type of training indices to use. Options include "random", "triad", "evmutation", "msatransformer", or "sim". "random" means simulations using random training data will be performed; "triad", "evmutation", and "msatransformer" all mean simulations using training data derived from zero-shot predictors will be performed; "sim" means simulations using artificially fitness-inflated training data will be performed. All options but "random" require additional information provided by the `training_specifics` keyword argument. |
| training_samples | Required Argument | The amount of training data to use. Options include "384", "48", or "24". |
| models_used | Required Argument | Options include "CPU", "GPU", "LimitedSmall", "LimitedLarge", and "XGB". Each option means simulations will be performed using models defined in a different MLDE parameter file. "CPU" will launch simulations run using all CPU-based models. "GPU" will launch simulations run using all GPU-based models. "XGB" will launch simulations run using all XGBoost-based models. "LimitedSmall" will launch all CPU-based models except for the sklearn ARDRegression, sklearn BaggingRegressor, and sklearn KNeighborsRegressor models -- these were the models omitted for simulations run using high-dimensional encodings with a small training set size. "LimitedLarge" will launch all CPU-based models except for the sklearn RandomForestRegressor, sklearn BaggingRegressor, and sklearn KNeighborsRegressor models -- these were the models omitted for simulations run using high-dimensional encodings with a large training set size. Simulations run using all options except "GPU" will spread across all available processors by default. Simulations run using "GPU" can only be run on a single thread. We provide "XGB" as a separate implementation as XGBoost models can require high levels of system RAM for larger encodings. For completing simulations efficiently, it can often be best to run XGBoost models independently on less cores (to save on RAM) and then perform all other CPU-based simulations across all available cores -- results can later be concatenated for processing. |
| saveloc | Required Argument | Where to save results |
| --training_specifics | Required Keyword Argument for All but `training_type = random` | This argument determines the sampling threshold to use when making simulations with zero-shot or artificially-fitness-inflated derived data. When `training_type` is "triad", "evmutation", or "msatransformer", options are "1600", "3200", "6400", "9600", "12800", "16000", or "32000". When `training_type` is "sim", options are "0", "0.1", "0.3", "0.5", or "0.7", which are the unnormalized fitness values of the thresholds used for the design of artifically fitness-inflated training data (i.e., 0.1 = 0.011 in our paper, 0.3 = 0.034, etc.) |
| --sim_low | Optional Keyword Argument | The simulation index (0-indexed) to start at, inclusive. By default, this is "0". |
| --sim_high | Optional Keyword Argument | The simulation index (0-indexed) to end at, exclusive. By default, this is "2000". |
| --n_jobs | Optional Keyword Argument | The number of CPUs to split over. This is ignored when `models_used` is "GPU". By default, all available cores on the machine are used. |
| --device | Optional Keyword Argument | The GPU index to run simulations on. This is only used when `models_used` is "GPU". |

A single run of `simulate_mlde.py` will generate a single time-stamped folder. The time-stamp format is "YYYYMMDD-HHMMSS" (Y = year, M = month, D = day, H = 24-hour, M = minute, S = second). The output folder contains different levels of folders describing the different arguments passed to the script. The first level pertains to `training_type`, the second to `--training_specifics`, the third to `encoding`, the fourth to `models_used`, and the fifth to `training_samples`. When `training_type` is "random", the `--training_specifics` layer is labeled as "random" as well -- all other options for `--training_specifics` will give the number pertaining to the sampling threshold. The fifth (`training_samples`) layer contains folders with the outputs of every simulation run (with the folder name corresponding to the simulation index) as well as a log file. The log file gives details on the inputs to the program. Each simulation folder contains the files "PredictionResults.csv", "SortedIndividualPreds.npy", and "SummaryResults.csv". The "PredictionResults.csv" file is equivalent to "PredictedFitness.csv" from [Outputs for execute_mlde.py](#outputs-for-execute_mldepy) with only the best model used for predictions; the "SortedIndividualPreds.npy" file is equivalent to "IndividualPreds.npy" from [Outputs for execute_mlde.py](#outputs-for-execute_mldepy); the "SummaryResults.csv" file is equivalent to LossSummaries.csv from [Outputs for execute_mlde.py](#outputs-for-execute_mldepy). The folder architecture may seem strange at first, but is designed such that results from many different simulation conditions can be rapidly combined into a single folder structure (e.g., by using the `rsync` command to synchronize all levels). The output files of all simulations can be further processed to calculate the summary metrics given in the two csv files in the CaltechData download folder.

A final note: some of the sklearn models used in MLDE are not always the most stable when trained with one-hot encodings. You may see an occasional warning that either "sklearn-regressor-LassoLarsCV" failed due misaligned dimensions or else that "sklearn-regressor-ARDRegression" failed due to an "unrecoverable internal error". These warnings should be sporadic (i.e., if every simulation throws this warning, you have a problem) but are not cause for concern -- MLDE internally handles failed models and drops them from analysis (to be specific, it assigns the failed model class an infinite cross-validation error, so unless you are averaging all model architectures in the ensemble you will not have a problem).

# Program Details
The MLDE algorithm takes as input all encodings corresponding to the combinations of amino acids found in the training data along with their measured fitness values. During the training stage, these sampled combinations are used to train a version of all inbuilt model architectures. Specifically, k-fold cross validation is performed to train each model using the default model parameters; mean validation error from the k-fold cross validation (mean squared error) is recorded for each architecture. Notably, all model instances trained during k-fold cross validation are also stored for later use. For instance, if evaluating all 22 inbuilt model architectures with 5-fold cross validation, 22 x 5 = 110 total trained model instances are recorded. In the next stage, H rounds of Bayesian hyperparameter optimization using the hyperopt Python package are optionally performed. Hyperparameters that minimize mean validation error are recorded. Post hyperparameter optimization, models are retrained using their optimal hyperparameters and mean validation error is recorded.

For making predictions, the top N model architectures (those with the lowest cross-validation error) are first identified. For each of the top N model architectures, predictions are made on the unsampled combinations by averaging the predictions of the kN model instances stored during cross validation. For instance, if testing the top 3 model architectures identified from 5-fold cross-validation, this means that the predictions of 3 x 5 = 15 total models (3 architectures x 5 model instances/architecture saved during cross validation) are used for prediction.

## Inbuilt Models
Currently, the prediction stage of MLDE can only be run using its inbuilt models. All models are either written in/derived from Keras, XGBoost, and scikit-learn. The models are detailed in the supporting information section of the paper accompanying this repository.

# Dependencies
## OS
MLDE was developed and vetted (using pytest) on a system running Ubuntu 18.04. In its current state it should run on any UNIX OS, but has not been tested on (nor can be expected to run) on Windows OS.

## Hardware
MLDE can be run on any standard setup (laptop or desktop) that has both CPU and GPU support. Some models for embedding generation  can require high GPU RAM usage, but should fit on decently sized commercial GPUs at a low enough batch size. During testing, we were able to generate all encodings for GB1 on an NVIDIA RTX 2070 (8 GB RAM) with room to spare (though it could be quite slow).

## Software
### MLDE Software
MLDE requires the dependencies given below. Instructions on the [tape-neurips](https://github.com/songlab-cal/tape-neurips2019) repository can be followed for its installation. Note that [EVcouplings](https://github.com/debbiemarkslab/EVcouplings) should be installed using `pip`. There is a clash between the CUDA Toolkit requirements for tensorflow-gpu v1.13.1 and Pytorch >v1.5: Given your CUDA Toolkit version installed, either Pytorch or tensorflow-gpu will not work.

  - python=3.7.3
  - numpy=1.16.4
  - pandas=0.25.3
  - tqdm=4.32.1
  - biopython=1.74
  - hyperopt=0.2.2
  - scipy=1.3.0
  - scikit-learn=0.21.2
  - tensorflow-gpu=1.13.1
  - keras=2.2.5
  - xgboost=0.90
  - nltk=3.4.4
  - psutil
  - pip
  - pytorch>=1.5
  - torchvision
  - torchaudio
  - cudatoolkit
  - transformers

  - evcouplings

Any specific versions listed were those used during the development of MLDE. There should be some leeway if users use different versions, though if running in a new environment, it is strongly recommended to perform the [pytest validation](#Installation-Validation) first.

### DeepSequence
DeepSequence requires the dependencies given below:

  - python=2.7
  - theano=1.0.1
  - cudatoolkit=10.1
  - cudnn
  - biopython
  - pandas
  - backports.functools_lru_cache

# Citing this Repository
Please cite our work "[Informed training set design enables efficient machine learning-assisted directed protein evolution](https://doi.org/10.1016/j.cels.2021.07.008)" when referencing this repository.

# Citing Supporting Repositories
MLDE relies on previously published GitHub repositories for encoding generation and zero-shot prediction. Links to the source repositories and bibtex entries for the accompanying works are below:

For the ESM module used during encoding and masking filling ([GitHub](https://github.com/facebookresearch/esm)):

```bibtex
@article{Rives2021,
author = {Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C Lawrence and Ma, Jerry and Fergus, Rob},
doi = {10.1073/pnas.2016239118},
journal = {Proceedings of the National Academy of Sciences},
title = {{Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences}},
url = {http://www.pnas.org/lookup/doi/10.1073/pnas.2016239118},
volume = {118},
year = {2021}
}

}

```

For the MSA Transformer used during encoding and mask filling ([GitHub](https://github.com/facebookresearch/esm)):

```bibtex
@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858},
  journal={bioRxiv}
}
```

For ProtBert and ProtBert-BFD used during encoding and mask filling ([GitHub](https://github.com/agemagician/ProtTrans)):

```bibtex
@article {Elnaggar2020.07.12.199554,
	author = {Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and BHOWMIK, DEBSINDHU and Rost, Burkhard},
	title = {ProtTrans: Towards Cracking the Language of Life{\textquoteright}s Code Through Self-Supervised Deep Learning and High Performance Computing},
	elocation-id = {2020.07.12.199554},
	year = {2020},
	doi = {10.1101/2020.07.12.199554},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554},
	eprint = {https://www.biorxiv.org/content/early/2020/07/21/2020.07.12.199554.full.pdf},
	journal = {bioRxiv}
}
```

For DeepSequence, used for zero-shot prediction ([GitHub](https://github.com/debbiemarkslab/DeepSequence)):

```bibtex
@article{Riesselman2018,
author = {Riesselman, Adam J and Ingraham, John B and Marks, Debora S},
doi = {10.1038/s41592-018-0138-4},
journal = {Nature Methods},
pages = {816--822},
title = {Deep generative models of genetic variation capture the effects of mutations},
url = {http://dx.doi.org/10.1038/s41592-018-0138-4 http://www.nature.com/articles/s41592-018-0138-4},
volume = {15},
year = {2018}
}
```

For the EVcouplings webapp, used for zero-shot prediction ([GitHub](https://github.com/debbiemarkslab/EVcouplings)):

```bibtex
@article{Hopf2019,
author = {Hopf, Thomas A. and Green, Anna G. and Schubert, Benjamin and Mersmann, Sophia and Sch{\"{a}}rfe, Charlotta P I and Ingraham, John B. and Toth-Petroczy, Agnes and Brock, Kelly and Riesselman, Adam J. and Palmedo, Perry and Kang, Chan and Sheridan, Robert and Draizen, Eli J. and Dallago, Christian and Sander, Chris and Marks, Debora S.},
doi = {10.1093/bioinformatics/bty862},
journal = {Bioinformatics},
pages = {1582--1584},
title = {The EVcouplings Python framework for coevolutionary sequence analysis},
url = {https://academic.oup.com/bioinformatics/article/35/9/1582/5124274},
volume = {35},
year = {2019}
}
```

For EVmutation, used for zero-shot prediction ([GitHub](https://github.com/debbiemarkslab/EVmutation)):

```bibtex
@article{Hopf2017,
author = {Hopf, Thomas A. and Ingraham, John B. and Poelwijk, Frank J. and Sch{\"{a}}rfe, Charlotta P.I. and Springer, Michael and Sander, Chris and Marks, Debora S.},
doi = {10.1038/nbt.3769},
journal = {Nature Biotechnology},
pages = {128--135},
title = {Mutation effects predicted from sequence co-variation},
url = {http://dx.doi.org/10.1038/nbt.3769},
volume = {35},
year = {2017}
}
```
