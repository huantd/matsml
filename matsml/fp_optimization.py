""" 
    Huan Tran (huantd@gmail.com)
    Fingerprint optimization

"""

import numpy as np
import pandas as pd
import warnings


from matsml.data import ProcessData
from matsml.io import AtomicStructure, get_key, goodbye, progress_bar
import os
import math
import sys


class FPOptimization:
    """ 
    Fingerprint: compute materials fingerprints from atomic structures
      - data_params: Dictionary, containing parameters needed, see manual

    """

    def __init__(self, data_params, optim_params):

        # Threadhold of fingerprint components
        self.data_params = data_params
        self.optim_params = optim_params

        self.data_file = get_key('data_file', self.data_params, None)
        if isinstance(self.data_file, type(None)):
            raise ValueError('  ERROR: data_file required')

        self.optim_type = get_key(
            'optim_type', self.optim_params, 'gradient_boosting')
        self.optim_fp_file = get_key(
            'optim_fp_file', self.optim_params, 'optim_fp.csv')

        data_processor = ProcessData(data_params=self.data_params)
        self.data_dict = data_processor.load_data()

        self.id_col = self.data_dict['id_col']
        self.x_cols = self.data_dict['x_cols']
        self.y_cols = self.data_dict['y_cols']
        self.train_set = self.data_dict['train_set'].reset_index()
        self.x_train = np.array(self.train_set[self.x_cols]).astype(np.float32)
        self.y_train = np.array(
            self.train_set[self.y_cols[0]]).astype(np.float32)

        print('  Fingerprint optimization ')
        print('    data_file'.ljust(32), self.data_file)
        print('    optim_type'.ljust(32), self.optim_type)

    def optimize(self):
        """
        Functionality: 
            Wrapper, compute materials fingerprints

        """

        if self.optim_type == 'gradient_boosting':
            optim_fp = self.optimize_gb()

        # Remove constant columns
        optim_fp = optim_fp.loc[:, (optim_fp != optim_fp.iloc[0]).any()]

        # Save
        optim_fp.to_csv(self.optim_fp_file, index=False)

        print('  Done fingerprint optimization, results saved in %s' %
              (self.optim_fp_file))

    def optimize_gb(self):
        """
        Functionality: 

        """
        from sklearn.model_selection import KFold
        from sklearn.ensemble import GradientBoostingRegressor

        data_fp = pd.read_csv(self.data_file, delimiter=',', header=0,
                              low_memory=False)

        print("Optimize the fingerprint using feature importance from gradient boosting")

        data_processor = ProcessData(data_params=self.data_params)

        # kfold splitting
        kf_ = KFold(n_splits=100, shuffle=True)
        kf = kf_.split(self.train_set)

        gbr = GradientBoostingRegressor(loss='squared_error', criterion='squared_error',
                                        n_estimators=int(0.27 * len(data_fp)))

        templ = "    cv, n_fps: {0:d} {1:d}"

        print('  Training model w/ cross validation')
        sel_xcols = []
        ncv = 0
        for train_cv, test_cv in kf:

            x_cv_train, x_cv_test, y_cv_train, y_cv_test = data_processor\
                .get_cv_datasets(self.train_set, self.x_cols, self.y_cols,
                                 train_cv, test_cv)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gbr.fit(x_cv_train, y_cv_train)

            importances = list(gbr.feature_importances_)

            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 4)) for
                                   feature, importance in zip(self.x_cols, importances)]

            # Sort the feature importances by most important first
            feature_importances = sorted(
                feature_importances, key=lambda x: x[1], reverse=True)

            feature_importances_pd = pd.DataFrame(feature_importances,
                                                  columns=['feature', 'importance'])

            # Select those summing up to at least 99% of importance
            feature_importances_cum = np.cumsum(
                np.array(feature_importances_pd['importance']))
            idx_up = [n for n, i in enumerate(
                feature_importances_cum) if i > 0.99][0]
            sel_features = feature_importances_pd.iloc[range(idx_up)]

            # Mearge to the previously selected fingerprints
            sel_xcols = list(set(sel_xcols).union(
                set(list(sel_features['feature']))))

            print(templ.format(ncv, len(list(sel_features['feature']))))
            ncv = ncv + 1

        print('FINAL', len(sel_xcols))

        sel_cols = self.id_col + self.y_cols + sel_xcols
        optim_fp = data_fp[sel_cols]

        return optim_fp
