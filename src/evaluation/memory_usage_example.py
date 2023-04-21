# importing the library
from memory_profiler import profile, memory_usage

import numpy as np
import pandas as pd

import logging

# create logger
logger = logging.getLogger('memory_profile_log')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler("memory_profile.log")
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)

from memory_profiler import LogFile
import sys

sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=True)


# instantiating the decorator
@profile
def app():

    for _ in range(5):

        temp = np.load('data/raw/npy/training.npy')
        a = np.load("data/raw/npy/training.npy")

        del a

    return temp

if __name__ == '__main__':
    temp = app()
