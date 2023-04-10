Project on Algorithmic Fairness
==============================

Special course in Responsible AI @ DTU


## Setup

Clone the repository and create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

### Create environment
Run the following:

```
conda create -n fairness_ai python=3.10
```

Install the dependencies:
```
pip install -r requirements.txt
```

#### PyTorch - CPU
If running on CPU install Pytorch with the following command:

```
pip3 install torch torchvision torchaudio
```

#### PyTorch - GPU (CUDA 11.6)
If running on GPU with CUDA 11.6 install Pytorch with the following command:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── catalan_data_course            <- The original, immutable data dump.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── __init__.py
    │   │   └── dataloader.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── __init__.py
    │       ├── model.py
    │       └── train_model.py
    │
    └── requirements.txt 
