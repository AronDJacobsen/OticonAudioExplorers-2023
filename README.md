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

## Project Organization
------------

    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   ├── raw            
    │   |   ├── mat                 <- Matlab data files
    │   │   └── npy                 <- Numpy data files
    │   └── processed               <- Nothing so far...
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
    │   └── models                  <- Scripts to evaluate model performance
    │       └── __init__.py      
    │        
    ├── .gitignore                  <- gitignore file
    └── requirements.txt            <- Depedencies...
