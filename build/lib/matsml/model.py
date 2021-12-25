# Copyright Huan Tran (huantd@gmail.com), 2021

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
import pandas as pd
from matsml.data import ProcessData

class ANN:

    def __init__(self,data_params,model_params):
        self.data_params = data_params
        self.model_params = model_params

    def train(self):
        
        # Work with data first
        data = ProcessData(data_params=self.data_params)

        # Extract x and y data from input
        data.get_xy()
        
        # Scale x and y using the methods specified in the input
        data.scale_xy()
        
        # Prepare train and test sets
        data.prepare_train_test_data()

        print ('Come here')

