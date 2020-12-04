MLDE
====
A machine-learning package for navigating combinatorial protein fitness landscapes. This repository accompanies our work "[Machine Learning-Assisted Directed Evolution Navigates a Combinatorial Epistatic Fitness Landscape with Minimal Screening Burden](#placeholder)". 

Table of Contents
-----------------
- [MLDE](#mlde)
  - [Table of Contents](#table-of-contents)
- [MLDE Concept](#mlde-concept)
- [Installation](#installation)
  - [Installation Validation](#installation-validation)
    - [Basic Tests](#basic-tests)
    - [Pytest Validation](#pytest-validation)
- [General Use](#general-use)
  - [Generating Encodings with GenerateEncodings.py](#generating-encodings-with-generateencodings.py)
    - [Inputs for GenerateEncodings.py](#inputs-for-generateencodings.py)
    - [Examples for GenerateEncodings.py](#examples-for-generateencodings.py)
    - [Outputs for GenerateEncodings.py](#outputs-for-generateencodings.py)
  - [Making Predictions with ExecuteMlde.py](#making-predictions-with-executemlde.py)
    - [Inputs for ExecuteMlde.py](#inputs-for-executemlde.py)
      - [TrainingData.csv](#training-data.csv)
      - [Custom Encodings](#custom-encodings)
      - [MldeParameters.csv](#mldeparameters.csv)
    - [Examples for ExecuteMlde.py](#examples-for-executemlde.py)
    - [Outputs for ExecuteMlde.py](#outputs-for-executemlde.py)
- [Program Details](#program-details)
  - [Inbuilt Models](#inbuilt-models)
- [Dependencies](#dependencies)
  - [OS](#os)
  - [Hardware](#hardware)
  - [Software](#software)
- [Citing this Repository](#citing-this-repository)

# MLDE Concept
MLDE attempts to learn a function that maps protein sequence to protein fitness for a combinatorial design space. The procedure begins by experimentally measuring the fitness values of a small subsample from a combinatorial library. These labeled variants are then used to train an ensemble of 22 models with varied architectures (see [Inbuilt Models](#inbuilt-models)). Trained models are next ranked by validation error, and the predictions of the N-best models are averaged to predict fitness values for the unsampled (“unlabeled”) variants. Unlabeled variants are ranked according to predicted fitness and returned. More detailed information on programmatic implementation is given [below](#program-details).

# Installation
1. For encoding generation, MLDE relies on the module [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019), which accompanies the work by Rao *et al.* *Evaluating Protein Transfer Learning with TAPE*. To include tape-neurips2019 as a submodule, the MLDE repository must be cloned with the "--recurse-submodules" flag as below. MLDE may not work correctly (or at all) if tape-neurips2019 is not included as a submodule. 

```
git clone --recurse-submodules https://github.com/fhalab/MLDE.git
```

2. The repository will be downloaded with the mlde.yml anaconda environment template in the top-level directory. It is highly recommended that MLDE be run from within the "mlde" conda environment. Note that this environment assumes you have pre-installed CUDA and have a CUDA-capable GPU on your system. If you would rather work in a separate environment, dependencies for MLDE can be found [here](#Dependencies). To build the mlde conda environment, run

```
cd ./MLDE
conda env create -f mlde.yml
```

3. Whatever environment you are working in, the next step is to complete installation of tape-neurips2019: Activate your desired conda environment, then navigate to the tape-neurips2019 submodule and install tape-neurips2019 dependencies (substitute "mlde" for whatever environment you will be working in):

```
conda activate mlde
cd ./Support/tape-neurips2019
pip install -e .
```

4. Still within the tape-neurips2019 submodule, download the model weights needed for generating learned embeddings by by running 

```bash
./download_pretrained_models.sh
```

## Installation Validation
### Basic Tests
Basic functionality of MLDE can be tested by running GenerateEncodings.py and ExecuteMlde.py with provided example data. Test command line calls are given below. To run GenerateEncodings.py:

```bash
conda activate mlde
python GenerateEncodings.py transformer GB1_T2Q --fasta 
  ./Validation/BasicTestData/2GI9.fasta --positions V39 D40 --batches 1
```

To run ExecuteMlde.py:

```bash
conda activate mlde
python ExecuteMlde.py ./Validation/BasicTestData/InputValidationData.csv 
  ./Validation/BasicTestData/GB1_T2Q_georgiev_Normalized.npy 
  ./Validation/BasicTestData/GB1_T2Q_ComboToIndex.pkl 
  --model_params ./Validation/BasicTestData/TestMldeParams.csv 
  --hyperopt
```

### Pytest Validation
MLDE has been thoroughly tested using [pytest](https://docs.pytest.org/en/stable/). These tests can be repeated as follows: activate the mlde conda environment, navigate to the top-level MLDE directory, and run pytest as below

```bash
conda activate mlde
cd ~/GitRepos/MLDE
PYTHONPATH=\$pathToRepoLocation/MLDE/ pytest
```

The cd command should be modified for wherever your copy of MLDE is saved. The PYTHONPATH argument should also be modified as appropriate based on the location of MLDE on your computer. Running the pytest tests will take multiple hours, so it is recommended to run them overnight. 

# General Use
MLDE works in two stages: Encoding generation and prediction. The encoding generation stage takes a parent protein sequence and generates all encodings for a given combinatorial library. The prediction stage uses the encodings from the first stage in combination with fitness data for a small set of combinations to make predictions on the fitness of the full combinatorial space. The MLDE package can generate all encodings presented in our accompanying paper; however, users can also pass in their own encodings (e.g. from a custom encoder) to the prediction stage scripts.

## Generating Encodings with GenerateEncodings.py
Encodings for MLDE are generated with the GenerateEncodings.py script. The combinatorial space can be encoded using one-hot, Georgiev, or learned embeddings. Learned embeddings can be generated from any of the completely unsupervised pre-trained models found in [tape-neurips2019](https://github.com/songlab-cal/tape-neurips2019#pretrained-models).

### Inputs for GenerateEncodings.py
There are a number of required and optional arguments that can be passed to GenerateEncodings.py, each of which are detailed in the table below

| Argument | Type | Description |
|:---------|---------------|-------------|
| encoding | Required Argument | This argument sets the encoding type used. Choices include "onehot", "georgiev", "resnet", "bepler", "unirep", "transformer", or "lstm".|
| name | Required Argument | Nickname for the protein. Will be used to prefix output files.|
| fasta | Required Argument for Learned Embeddings | The parent protein amino acid sequence in fasta file format.|
| positions | Required Argument for Learned Embeddings | The positons and parent amino acids to include in the combinatorial library. Input format must be "AA## AA##" and should be in numerical order of the positions. For instance, to mutate positions Q23 and F97 in combination, the flag would be written as `--positions Q23 F97`.|
| n_combined | Required Argument for Georgiev or Onehot Encodings | The number of positions in the combinatorial space. |
| output | Optional Argument | Output location for saving data. Default is the current working directory if not specified. |
| batches | Optional Argument | Generating the largest embedding spaces can require high levels of system RAM. This parameter dictates how many batches to split a job into. If not specified, the program will attempt to automatically determine the appropriate number of batches given the available RAM on your system. |

### Examples for GenerateEncodings.py
The below example is a command line call for generating Georgiev encodings for a 4-site combinatorial space (160000 total variants). This command could also be used for generating onehot encodings. Note that, because georgiev and onehot encodings are context-independent, no fasta file is needed. Indeed, the encodings generated from the below call could be applied to any 4-site combinatorial space.

```bash
python GenerateEncodings.py georgiev example_protein --n_combined 4 
```

The below example is a command line call for generating transformer embeddings for the 4-site GB1 combinatorial library discussed in our work. Note that this command could be used for generating any of the learned embeddings (with appropriate substitution of the `encoding` argument). Because embeddings are context-aware, these encodings should not be used for another protein.

```bash
python GenerateEncodings.py transformer GB1_T2Q 
  --fasta .Validation/BasicTestData/2GI9.fasta
  --positions V39 D40 G41 V54 --batches 4
```

The input fasta file looks as below:

```
>GB1_T2Q
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE
```

### Outputs for GenerateEncodings.py
Every run of GenerateEncodings.py produces a time-stamped folder containing all results. The time-stamp format is "YYYYMMDD-HHMMSS" (Y = year, M = month, D = day, H = 24-hour, M = minute, S = second). The time-stamped folder contains subfolders "Encodings" and "Fastas". 

The "Encodings" folder will contain the below files ("\$NAME" is from the `name` argument of GenerateEncodings.py; "\$ENCODING" is from the `encoding` argument):

| Filename | Description |
|:---------|-------------|
|\$NAME_\$ENCODING_Normalized.npy| Numpy array containing the mean-centered, unit-scaled amino acid embeddings. These are the embeddings that will typically be used for generating predictions, and take the shape $20^C x C x L$, where $C$ is the number of amino acid positions combined and $L$ is the number of latent dimensions per amino acid for the encoding.|
|\$NAME_\$ENCODING_UnNormalized.npy| Numpy array containing the unnormalized amino acid embeddings. This tensor will take the same shape as \$NAME_\$ENCODING_Normalized.npy.|
|\$NAME_\$ENCODING_ComboToIndex.pkl| A pickle file containing a dictionary linking amino acid combination to the index of that combination in the output encoding tensors. Note that combinations are reported in order of amino acid index (e.g. a combination of A14, C23, Y60, and W91 would be written as "ACYW").|
|\$NAME_\$ENCODING_IndexToCombo.pkl| A pickle file containing a dictionary that relates index in the encoding tensor to the combination.|

Note that when encoding is "onehot", only unnormalized embeddings will be returned.

The "Fastas" directory is only populated when one of the learned embeddings is generated. It contains fasta files with all sequences used to generated embeddings, split into batches as appropriate.

## Making Predictions with ExecuteMlde.py
MLDE predictions are made using the ExecuteMlde.py script. Inputs to this script include the encodings for all possible members of the combinatorial space, experimentally determined sequence-function data for a small number of combinations (for use as training data), and a dictionary linking all possible members of the combinatorial space to their associated encoding. 

### Inputs for ExecuteMlde.py
| Argument | Type | Description |
|:---------|-----------|-------------|
| training_data | Required Argument | A csv file containing the sequence-function information for sampled combinations. More information on this file can be found [below](#trainingdata.csv). |
| encoding_data | Required Argument | A numpy array containing the embedding information for the full combinatorial space. Encoding arrays generated by GenerateEncodings.py can be passed directly in here. Custom encodings can be passed in here too, the details of which are discussed [below](#custom-encodings). |
| combo_to_ind_dict | Required Argument | A pickle file containing a dictionary that links a combination to its index. The ComboToIndex.pkl file output by GenerateEncodings.py can be passed in directly here. |
| model_params | Optional Argument | A csv file dictating which inbuilt MLDE models to use as well as how many rounds of hyperparameter optimization to perform. The makeup of this file is discussed [below](#mldeparameters.csv). |
| output | Optional Argument | The location to save the results. Default is the current working directory. |
| n_averaged | Optional Argument | The number of top-performing models to average to get final prediction results. Default is 3 |
| n_cv | Optional Argument | The number of rounds of cross validation to perform during training. Default is 5. |
| no_shuffle | Flag | When set, the indices of the training data will **not** be shuffled for cross-validation. Default is to shuffle indices. Note that there may be benefit in running MLDE multiple times, then averaging the outcome of all runs; this ensures that results are not adversely affected by a randomly chosen cross-validation split.|
| hyperopt | Flag | When set, hyperparameter optimization will also be performed. Note that this can greatly increase the run time of MLDE depending on the models included in the run. The default is to not perform hyperparameter optimization. |

#### TrainingData.csv
This csv file contains the sequence-function data for the protein of interest. An example csv file can be found in MLDE/Validation/BasicTestData/InputValidationData.csv. The top few rows of this file are demonstrated below:

| AACombo | Fitness |
|:--------|---------|
| CCCC | 0.5451 |
| WCPC | 0.0111 |
| WPGC | 0.0097 |
| WGPP | 0.0022 |

The two column headers must always be present and always have the same name. Sequence is input as the combination identity, which is the amino acid present at each position in order. For instance, a combination of A14, C23, Y60, and W91 would be written as "ACYW". 

While not strictly required, it is recommended to normalize fitness in some manner. Common normalization factors would be the fitness of the parent protein or the maximum fitness in the training data. 

#### Custom Encodings
Any encoding can be passed into ExecuteMlde.py so long as it meets the dimensionality requirements. Specifically, the array must take the shape $20^C x C x L$, where $C$ is the number of amino acid positions combined and $L$ is the number of latent dimensions per amino acid for the encoding. The program will throw an exception if a non-3D encoding is passed in as the encoding_data argument.

Note that for all but the convolutional neural networks, the last 2 dimensions of the input space will be flattened before processing. In other words, convolutional networks are trained on 2D encodings and all other models on 1D encodings. 

#### MldeParameters.csv
This file details what models are included in an MLDE run and how many hyperparameter optimization rounds will be executed for each model. By default, the file found at MLDE/Support/Params/MldeParameters.csv is used, though users can pass in their own versions; the default file can be used as a template for custom parameter files, but should never be changed itself. The contents of the MldeParameters.csv file are copied below:

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

The column names should not be changed. Rows should never be deleted from this file. Changing the "Include" column contents to 'FALSE' will stop a model from being included in the ensemble trained for MLDE. The "NHyperopt" column contents can be changed to alter how many hyperparameter optimization rounds are performed when the `hyperopt` flag is thrown. Note that Keras-based models can take a long time for hyperparameter optimization, hence why only 10 rounds are performed by default.

### Examples for ExecuteMlde.py
The below is an example run of ExecuteMlde.py using information output by GenerateEncodings.py as its inputs. Note that all optional arguments are used here for demonstration purposes; they don't all need to be used in practice.

```bash
python ExecuteMlde.py ./FitnessInfo.csv ./GB1_T2Q_georgiev_Normalized.npy ./GB1_T2Q_ComboToIndex.pkl --model_params ./MyCustomModelParams.csv --output ~/Documents/MyMldeRun --n_averaged 5 --n_cv 10 --hyperopt
```

### Outputs for ExecuteMlde.py
Every run of ExecuteMlde.py produces a time-stamped folder containing all results. The time-stamp format is "YYYYMMDD-HHMMSS" (Y = year, M = month, D = day, H = 24-hour, M = minute, S = second). The time-stamped folder contains the files "PredictedFitness.csv", "LossSummaries.csv", "CompoundPreds.npy", "IndividualPreds.npy", and "PredictionStandardDeviation.npy". If hyperparameter optimization is performed, an additional file called "HyperoptInfo.csv" will also be generated. The contents of each file are detailed below:

| Filename | Description |
|:---------|-------------|
| PredictedFitness.csv | This csv file reports the average predicted fitness of the top models given by `n_averaged` for all possible combinations (including combinations in the training data). Whether a combination was present in the training data or not is clearly marked. |
| LossSummaries.csv | This csv file reports the cross-validation training and testing error of the best models from each class. |
| CompoundPreds.npy | This numpy file contains an array with shape $M x 20^C$, where $M$ is the number of models and $C$ is the number of amino acids in the combinatorial space. This array gives the average predictions of the top-M models for all possible combinations. For instance, index 0 gives the predictions of the best model; index 1 gives the average predictions of the top 2 models, and so on. |
| IndividualPreds.npy| This numpy array gives the predictions of all models, ordered by the model's cross-validation testing error. This array is the same shape as CompoundPreds.npy. |
| PredictionStandardDeviation.npy| This numpy array gives the standard deviation of predictions across the models generated from different cross-validation steps. It has the same shape and ordering as IndividualPreds.npy. |
|HyperoptInfo.csv | This csv file gives details on the hyperparameter optimization procedure, including parameter values tested and associated cross-validation errors in each iteration. |

# Program Details
The MLDE algorithm takes as input all encodings corresponding to the combinations of amino acids found in the training data along with their measured fitness values. During the training stage, these sampled combinations are used to train a version of all inbuilt models using K-fold cross validation and the default model parameters; mean validation error from the K-fold cross validation is recorded. Against using K-fold validation, in the next stage, H rounds of Bayesian hyperparameter optimization using the hyperopt Python package are optionally performed. Hyperparameters which minimize mean validation error are recorded. Post hyperparameter optimization, models are retrained using their optimal hyperparameters and mean validation error is recorded. 

For making predictions, the top-N model architectures (those with the lowest cross-validation error after hyperparameter optimization) are first identified. For each of the top-N model architectures, predictions are made on the unsampled combinations by averaging the predictions of the K models trained during cross validation. The predictions made by each of the N models are then averaged to return a single final prediction of the unsampled values. In total, this means that K x N models are averaged to generate a single prediction for MLDE (K from each of the top-N model architectures).

## Inbuilt Models
Currently, the prediction stage of MLDE can only be run using its inbuilt models, though it could potentially be expanded to allow custom designation of models in the future. All models are either written in/derived from Keras, XGBoost, and scikit-learn. The models are detailed in the supporting information section of the paper accompanying this repository.

# Dependencies
## OS
MLDE was developed and vetted (using pytest) on a system running Ubuntu 18.04. In its current state it should run on any UNIX OS, but has not been tested on (nor can be expected to run) on Windows OS. 

## Hardware
MLDE can be run on any standard setup (laptop or desktop) that has both CPU and GPU support. The lstm and unirep models can require up to 8 GB GPU RAM for encoding generation, while the others are fairly tame and should fit on most GPUs; encodings were generated with NVIDIA RTX2070 and TitanV GPUs during development of this software.

## Software
MLDE requires the dependencies given below. The submodule [tape-neurips](https://github.com/songlab-cal/tape-neurips2019) submodule has its own dependencies that will be installed during its setup.

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

Specific versions used during the development of MLDE are listed here. MLDE is validated to be stable using these versions, though there should be some leeway if users use different versions. If running in a new environment, it is strongly recommended to perform the [pytest validation](#Installation-Validation) first.

# Citing this Repository
Please cite our work _____ when referencing this repository.
