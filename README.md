Oticon Audio Explorers - 2023
==============================

Case competition - Electrical Challenge Option 1

## Setup

Clone the repository and create a virtual environment (with Python 3.10).

### Create environment
Run the following:

```
conda create -n oticon python=3.10
```

Install the dependencies:
```
pip install -r requirements.txt
```

#### GPU - PyTorch installation

The code in this repository have been run with the following `torch`-version:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

## Reproducibility

The presented solution requires running two things for being reproducible:
1) Model training
2) Pruning loop

When these have been run, `src/evaluation/model_evaluation.ipynb` notebook can be used for obtaining the results presented in the associated project report.

#### Model training

For training the model, one should run the following command from the root of the repository:

```
python src/model/train.py
```

The optimized hyperparameter values are specified in the bottom of the file and can be adjusted if one wants to explore other parameter combinations. 

#### Pruning loop

For running the pruning loop on a pre-trained model, one should run:

```
python src/evaluation/pruning.py
```

This saves information on batch-wise model performance for a variety of pruning ratios in a .csv-file in the `results`-folder. For running multiple realizations, one should change the `realization_number` parameter that is specified in the bottom of the `pruning.py` file. 


## Project Organization
------------

    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   └── raw            
    │       ├── mat                 <- Matlab data files
    │       └── npy                 <- Numpy data files
    │
    ├── setup.py                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   |   ├── __init__.py
    │   |   ├── exploration.ipynb   <- Exploration of the dataset
    │   │   └── dataloader.py       <- Nothing so far...
    │   │
    │   ├── models                  <- Scripts to define and train models
    │   |   └── __init__.py
    │   │
    │   └── evaluation              <- Scripts to evaluate model performance
    │       └── __init__.py      
    │        
    ├── figures                     <- Contains figures that are presented in the associated report.
    ├── .gitignore                  <- gitignore file
    └── requirements.txt            <- Depedencies...
