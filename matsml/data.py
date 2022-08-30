""" Huan Tran (huantd@gmail.com)

    Data module: data related functionalities (pre/postprocessing) needed for 
      matsML.

"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from matsml.io import get_key, goodbye
import io
import os
import requests
import heapq
import random
from sklearn.metrics import mean_squared_error


class Datasets:
    """ 
    Retrieve some datasets made available at www.matsml.org
    """

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.kwargs = kwargs

        sum_url = 'http://www.matsml.org/data/datasets.csv'
        self.datasets = pd.read_csv(io.StringIO(requests.get(sum_url).content.
                                                decode('utf-8')))

    def summary(self):
        """ Show what datasets available """

        print(' Available datasets')
        print(self.datasets[['id', 'name']].to_string(index=False))

    def load_dataset(self):
        """ Load datasets from matsml.org by name """

        print('  Load requested dataset(s)')
        for dataset_name in self.kwargs.values():
            sel_row = self.datasets[self.datasets['name'] == dataset_name]

            if len(sel_row) > 0:
                data_url = np.array(sel_row['url']).astype(str)[0]
                fname = data_url.split('/')[-1]
                os.system('wget -O '+fname+' --no-check-certificate '+data_url)
                if fname.startswith('fp_'):
                    print('  Data saved in '+fname)
                else:
                    os.system('tar -xf '+fname)
                    print('  Data saved in '+dataset_name)
            else:
                raise ValueError('  ERROR: dataset '+str(dataset_name) +
                                 ' not found.')


class ProcessData:
    """ 
    Process data needed for learning
    """

    def __init__(self, data_params):
        self.data_params = data_params

        # List of ids that must be in training set
        self.train_ids = get_key('train_ids', self.data_params, [])

        # Default x_scaling and y_scaling is minmax
        self.x_scaling = get_key('x_scaling', self.data_params, 'minmax')
        self.y_scaling = get_key('y_scaling', self.data_params, 'minmax')

        # Default x_scaling and y_scaling is random
        self.sampling = get_key('sampling', self.data_params, 'random')

        # If we want to save the train/test spliting
        self.save_split = get_key('save_split', self.data_params, True)

    def read_data(self):
        print('  Read data')

        self.data_file = self.data_params['data_file']

        # ID column
        id_col = self.data_params['id_col']
        if len(id_col) != 1:
            raise ValueError('  ERROR: There must be one ID column only')
        else:
            if id_col[0] == 'index':
                raise ValueError(
                    '  ERROR: "index" reserved, dont use for ID column')
            else:
                self.id_col = id_col

        y_cols = self.data_params['y_cols']
        self.y_cols = y_cols

        y_dim = len(y_cols)
        self.y_dim = y_dim

        comment_cols = self.data_params['comment_cols']
        self.comment_cols = comment_cols

        data_fp = pd.read_csv(self.data_file, delimiter=',', header=0,
                              low_memory=False)

        self.data_size = len(data_fp)

        # No duplicate allowed in the ID column
        n_ids = len(list(set(list(data_fp[self.id_col[0]]))))
        if n_ids != self.data_size:
            raise ValueError('  ERROR: There are duplicates in the ID columns')

        # list of columns for x
        x_cols = [col for col in list(data_fp.columns) if col not in
                  (y_cols+id_col+comment_cols)]
        self.x_cols = x_cols

        # list of selector cols
        sel_cols = [col for col in x_cols if "selector" in col]
        self.sel_cols = sel_cols

        # values of sel cols
        sel_vals = data_fp[sel_cols].drop_duplicates()
        self.sel_vals = sel_vals

        # x and y data for learning
        self.y = data_fp[id_col + sel_cols + y_cols]
        self.x = data_fp[id_col + x_cols]

        # fill nan by 0 in fingerprint
        self.x = self.x.fillna(0)

        # ser ntrains at 0.8 if this key not presents
        self.n_trains = int(
            get_key('n_trains', self.data_params, 1.0) * self.data_size)
        self.n_tests = self.data_size - self.n_trains

        # Print some data parameters
        tpl = '{} {}{}{}{}'
        print('    data file'.ljust(32), self.data_file)
        print('    data size'.ljust(32), self.data_size)
        print('    training size'.ljust(32), round(
            100*self.n_trains/self.data_size, 1), ' %')
        print('    test size'.ljust(32), round(
            100*self.n_tests/self.data_size, 1), ' %')
        print('    x dimensionality'.ljust(32), len(self.x_cols))
        print('    y dimensionality'.ljust(32), len(self.y_cols))
        print('    y label(s)'.ljust(32), self.y_cols)

    def scale_x(self):
        """ Scale x before learning """
        import joblib

        xscaler_file = 'xscaler.pkl'

        x = self.x

        print('  Scaling x'.ljust(32), self.x_scaling)

        if self.x_scaling == 'minmax':
            self.xscaler = preprocessing.MinMaxScaler()
            x_scaled = self.xscaler.fit_transform(x.drop(self.id_col, axis=1))
            joblib.dump(self.xscaler, xscaler_file)
            print('    xscaler saved in'.ljust(32), xscaler_file)
        if self.x_scaling == 'normalize':
            self.xscaler = preprocessing.StandardScaler()
            x_scaled = self.xscaler.fit_transform(x.drop(self.id_col, axis=1))
            joblib.dump(self.xscaler, xscaler_file)
            print('    xscaler saved in'.ljust(32), xscaler_file)
        elif self.x_scaling == 'none':
            x_scaled = x.drop(self.id_col, axis=1)

        # Convert nparray back pandas and stack the ID column
        x_scaled_df = pd.DataFrame(x_scaled, columns = self.x_cols)
        x_scaled_df[self.id_col] = x[self.id_col]
        self.x_scaled = x_scaled_df

    def scale_y(self):
        """ 
        Scale y before learning 

          I want to handle both selectors and scaling here, so things are a 
          bit complicated. First, I must keep track the selector values so 
          the inverse scaling can be done. Then, I also want to keep track
          of the ID so the information of the predictions can be completed.
          Therefore, dont surprise of the following complicated lines.
          I will make it simpler latter. 

          Now, my assumptions for the selector vectors are
             (1) "Selector" columns are named "selector*"
             (2) Only one element of the selector vector is non-zero 
                 and it must be 1.
        """

        y = self.y
        y_cols = self.y_cols
        y_dim = self.y_dim
        sel_cols = self.sel_cols
        print('  Scaling y'.ljust(32), self.y_scaling)

        # Compute some distribution parameters and do scaling
        if len(sel_cols) > 0:
            y_mean = pd.DataFrame(columns=['sel']+y_cols)
            y_std = pd.DataFrame(columns=['sel']+y_cols)
            y_min = pd.DataFrame(columns=['sel']+y_cols)
            y_max = pd.DataFrame(columns=['sel']+y_cols)

            for sel in sel_cols:
                y_sel = y[y[sel] == 1]

                this_row = pd.DataFrame([np.mean(np.array(y_sel[col])) for col
                                         in y_cols], columns=y_cols)
                this_row['sel'] = sel
                y_mean = pd.concat([y_mean,this_row], axis = 0)

                this_row = pd.DataFrame([np.std(np.array(y_sel[col])) for col
                                         in y_cols], columns=y_cols)
                this_row['sel'] = sel
                y_std = pd.concat([y_std, this_row], axis = 0)

                this_row = pd.DataFrame([np.amin(np.array(y_sel[col])) for col
                                         in y_cols], columns=y_cols)
                this_row['sel'] = sel
                y_min = pd.concat([y_min, this_row], axis = 0)

                this_row = pd.DataFrame([np.amax(np.array(y_sel[col])) for col
                                         in y_cols], columns=y_cols)
                this_row['sel'] = sel
                y_max = pd.concat([y_max, this_row], axis = 0)

            self.y_mean = y_mean
            self.y_std = y_std
            self.y_min = y_min
            self.y_max = y_max

            y_scaled = pd.DataFrame(columns=self.id_col+sel_cols+y_cols)
            y_scaled[self.id_col] = y[self.id_col]
            y_scaled[sel_cols] = y[sel_cols]

            for i, j, sel in ((a, b, c) for a in range(len(y)) for b in
                              range(y_dim) for c in sel_cols):
                this_row = y.iloc[i]
                if this_row[sel].astype(int) == 1:
                    ymean = float(y_mean.loc[y_mean['sel'] == sel][y_cols[j]])
                    ystd = float(y_std.loc[y_std['sel'] == sel][y_cols[j]])
                    ymin = float(y_min.loc[y_min['sel'] == sel][y_cols[j]])
                    ymax = float(y_max.loc[y_max['sel'] == sel][y_cols[j]])
                    if str(self.y_scaling) == 'normalize':
                        y_scaled.at[i, y_cols[j]] = (y.at[i, y_cols[j]]-ymean) /\
                            ystd
                    elif str(self.y_scaling) == 'minmax':
                        y_scaled.at[i, y_cols[j]] = (y.at[i, y_cols[j]]-ymin) /\
                            (ymax-ymin)
                    elif str(self.y_scaling) == 'logpos':
                        y_scaled.at[i, y_cols[j]] = np.log(y.at[i, y_cols[j]])
                    elif str(self.y_scaling) == 'logfre':
                        y_scaled.at[i, y_cols[j]] = np.log(
                            y.at[i, y_cols[j]]-ymin+1)
                    elif str(self.y_scaling) == 'none':
                        y_scaled.at[i, y_cols[j]] = y.at[i, y_cols[j]]

            self.y_scaled = y_scaled

        elif len(sel_cols) == 0:
            self.y_mean = pd.DataFrame([np.mean(np.array(y[col])) for col
                                        in y_cols]).T
            self.y_mean.columns = y_cols

            self.y_std = pd.DataFrame([np.std(np.array(y[col])) for col
                                       in y_cols]).T
            self.y_std.columns = y_cols

            self.y_min = pd.DataFrame([np.amin(np.array(y[col])) for col
                                       in y_cols]).T
            self.y_min.columns = y_cols

            self.y_max = pd.DataFrame([np.amax(np.array(y[col])) for col
                                       in y_cols]).T
            self.y_max.columns = y_cols

            y_scaled = pd.DataFrame(columns=self.id_col+y_cols)
            for i, j in ((a, b) for a in range(len(y)) for b in range(y_dim)):
                y_scaled[self.id_col] = y[self.id_col]
                if str(self.y_scaling) == 'normalize':
                    delta_y = y.at[i, y_cols[j]]-self.y_mean.at[0, y_cols[j]]
                    y_scaled.at[i, y_cols[j]] = delta_y / \
                        self.y_std.at[0, y_cols[j]]
                elif str(self.y_scaling) == 'minmax':
                    delta_y = (self.y_max.at[0, y_cols[j]] -
                               self.y_min.at[0, y_cols[j]])
                    y_scaled.at[i, y_cols[j]] = (y.at[i, y_cols[j]] -
                                                 self.y_min.at[0, y_cols[j]])/delta_y
                elif str(self.y_scaling) == 'logpos':
                    y_scaled.at[i, y_cols[j]] = np.log(y.at[i, y_cols[j]])
                elif str(self.y_scaling) == 'logfre':
                    y_scaled.at[i, y_cols[j]] = np.log(
                        y.at[i, y_cols[j]]-self.y_min.at[0, y_cols[j]]+1)
                elif str(self.y_scaling) == 'none':
                    y_scaled.at[i, y_cols[j]] = y.at[i, y_cols[j]]

            self.y_scaled = y_scaled

    def split_train_test(self):
        """ 
        Prepare train and test sets using the sampling method specified. 
        'random' and 'stratified' currently available, more to come.
        """

        print('  Prepare train/test sets'.ljust(32), self.sampling)

        id_col = self.id_col
        scaled_data = self.scaled_data

        # IDs that must be in training set
        train_ids = self.train_ids

        # reduced data
        scaled_data_red = scaled_data[~scaled_data[id_col[0]].isin(train_ids)]

        # training set initialized
        train_set_ids = [idx for idx in train_ids]

        if self.sampling == 'random':
            for idx in scaled_data_red.sample(n=self.n_trains-len(train_ids),
                                              random_state=42)[id_col[0]].tolist():
                train_set_ids.append(idx)
            test_set_ids = [idx for idx in scaled_data[id_col[0]].tolist()
                            if idx not in train_set_ids]

        elif self.sampling == 'stratified':
            """ Stratified on a PCA-projected manifold"""

            from sklearn.decomposition import PCA
            from scipy.stats import binned_statistic_2d

            train_set_ids = []

            data = np.array(self.x_scaled[self.x_cols])
            pca = PCA(n_components=2)
            x_pca = pca.fit_transform(data)

            # Do some statistics on the PCA trasformed data
            statistic, xedges, yedges, binnumber = binned_statistic_2d(x_pca[:, 0],
                                                                       x_pca[:, 1], data, statistic='count', bins=5)

            # bin ID and number of entries in each bin
            bin_ids, bin_freqs = np.unique(binnumber, return_counts=True)

            # list of entries to be selected from each bin.
            b_dists = [min(max(1, round(bf*self.n_trains/self.data_size)), bf)
                       for bf in bin_freqs]

            # update self.n_trains. Need more work here to avoid this adjustment
            self.n_trains = sum(b_dists)

            for i in range(len(bin_ids)):
                train_set_ids = train_set_ids+random.sample(list(np.where(
                    binnumber == bin_ids[i])[0]), b_dists[i])

            test_set_ids = [idx for idx in scaled_data[id_col[0]].tolist()
                            if idx not in train_set_ids]

        else:
            raise ValueError('      ERROR: unavailable sampling')

        self.train_set = scaled_data[scaled_data[id_col[0]].isin(
            train_set_ids)]
        self.test_set = scaled_data[scaled_data[id_col[0]].isin(test_set_ids)]

        # Save train/test split
        if self.save_split:
            train_set_tmp = self.unscaled_data[self.unscaled_data[id_col[0]].isin(
                train_set_ids)]
            test_set_tmp = self.unscaled_data[self.unscaled_data[id_col[0]].isin(
                test_set_ids)]
            train_set_tmp.to_csv('train_data.csv', index=False)
            test_set_tmp.to_csv('test_data.csv', index=False)

    def invert_scale_y(self, y_scaled, data_dict, message):
        """ Unscale the y data """

        print('    unscaling y:', data_dict['y_scaling'])

        id_col = data_dict['id_col']
        y_cols = data_dict['y_cols']
        sel_cols = data_dict['sel_cols']
        y_scaling = data_dict['y_scaling']
        y_org = data_dict['y_org']
        y_mean = data_dict['y_mean']
        y_std = data_dict['y_std']
        y_min = data_dict['y_min']
        y_max = data_dict['y_max']
        y_md_cols = data_dict['y_md_cols']
        yerr_md_cols = data_dict['yerr_md_cols']

        y_org = y_org[id_col+sel_cols+y_cols]
        y_dim = len(y_cols)
        ids_list = list(y_scaled[id_col[0]])

        # Starting from y_scaled, unscale the data in y_cols, and add
        # to y_org, then select nonan to get y_unscaled
        #
        if len(sel_cols) > 0:                                # selecter columns
            for idn, jy, sel in ((a, b, c) for a in ids_list for b in
                                 range(y_dim) for c in sel_cols):
                idx0 = np.array(y_org[y_org[id_col[0]] == idn].index)[0]
                idx1 = np.array(y_scaled[y_scaled[id_col[0]] ==
                                         idn].index)[0]

                if y_scaled.at[idx1, sel] > 0.0:
                    if str(y_scaling) == 'minmax':
                        ymax = float(y_max.loc[y_max['sel'] ==
                                               sel][y_cols[jy]])
                        ymin = float(y_min.loc[y_min['sel'] ==
                                               sel][y_cols[jy]])
                        y_org.at[idx0, y_md_cols[jy]] = y_scaled.at[idx1,
                                                                    y_md_cols[jy]]*(ymax-ymin)+ymin
                    elif str(y_scaling) == 'normalize':
                        ymean = float(y_mean.loc[y_max['sel'] ==
                                                 sel][y_cols[jy]])
                        ystd = float(y_std.loc[y_min['sel'] ==
                                               sel][y_cols[jy]])
                        y_org.at[idx0, y_md_cols[jy]] = y_scaled.at[idx1,
                                                                    y_md_cols[jy]]*ystd + ymean
                    elif str(y_scaling) == 'logpos':
                        y_org.at[idx0, y_md_cols[jy]] = np.exp(
                            y_scaled.at[idx1, y_md_cols[jy]])
                    elif str(y_scaling) == 'logfre':
                        ymin = float(
                            y_min.loc[y_min['sel'] == sel][y_cols[jy]])
                        y_org.at[idx0, y_md_cols[jy]] = np.exp(
                            y_scaled.at[idx1, y_md_cols[jy]]) - 1 + ymin
                    elif str(y_scaling) == 'none':
                        y_org.at[idx0, y_md_cols[jy]] = y_scaled.at[idx1,
                                                                    y_md_cols[jy]]

            y_unscaled = y_org.dropna(subset=y_md_cols)

            # Get RMSE of each prop
            for sel in sel_cols:
                # more work needed here
                y_sel = y_unscaled.loc[y_org[sel] == 1]
                for y_col in y_cols:
                    this_rmse = np.sqrt(np.mean((np.array(y_sel[y_col]) -
                                                 np.array(y_sel['md_'+y_col]))**2))
                    print("      rmse", str(message).ljust(12), sel,
                          str(y_col).ljust(16), round(this_rmse, 6))

        elif len(sel_cols) == 0:                               # selecter columns
            for idn, jy in ((a, b) for a in ids_list for b in range(y_dim)):
                idx0 = np.array(y_org[y_org[id_col[0]] == idn].index)[0]
                idx1 = np.array(y_scaled[y_scaled[id_col[0]] ==
                                         idn].index)[0]
                if str(y_scaling) == 'minmax':
                    delta_y = (y_max.at[0, y_cols[jy]]-y_min.at[0, y_cols[jy]])
                    y_org.at[idx0, y_md_cols[jy]] = y_scaled.at[idx1,
                                                                y_md_cols[jy]]*delta_y+y_min.at[0, y_cols[jy]]
                    if len(yerr_md_cols) > 0:
                        y_org.at[idx0, yerr_md_cols[jy]] = y_scaled.at[idx1,
                                                                       yerr_md_cols[jy]]*delta_y
                elif str(y_scaling) == 'normalize':
                    y_org.at[idx0, y_md_cols[jy]] = (y_scaled.at[idx1,
                                                                 y_md_cols[jy]]*y_std.at[0, y_cols[jy]]) + \
                        y_mean.at[0, y_cols[jy]]
                    if len(yerr_md_cols) > 0:
                        y_org.at[idx0, yerr_md_cols[jy]] = y_scaled.at[idx1,
                                                                       yerr_md_cols[jy]]*y_std.at[0, y_cols[jy]]
                elif str(y_scaling) == 'logpos':
                    y_org.at[idx0, y_md_cols[jy]] = np.exp(y_scaled.at[idx1,
                                                                       y_md_cols[jy]])
                elif str(y_scaling) == 'logfre':
                    y_org.at[idx0, y_md_cols[jy]] = np.exp(y_scaled.at[idx1,
                                                                       y_md_cols[jy]]) - 1 + y_min.at[0, y_cols[jy]]
                elif str(y_scaling) == 'none':
                    y_org.at[idx0, y_md_cols[jy]] = y_scaled.at[idx1,
                                                                y_md_cols[jy]]
                    if len(yerr_md_cols) > 0:
                        y_org.at[idx0, yerr_md_cols[jy]] = y_scaled.at[idx1,
                                                                       yerr_md_cols[jy]]

            y_unscaled = y_org.dropna(subset=y_md_cols)

            for y_col in y_cols:                                      # rmse_cv
                this_rmse = np.sqrt(mean_squared_error(np.array(y_unscaled
                                                                [y_col]), np.array(y_unscaled['md_'+y_col])))
                print("       rmse", str(message).ljust(12), str(y_col).
                      ljust(16), round(this_rmse, 6))

        return y_unscaled

    def load_data(self):
        """ 
        Load data, scale data, and split data => train/test sets 
        """

        # Extract x and y data from input file
        self.read_data()

        # Scale x
        self.scale_x()

        # Scale y
        self.scale_y()

        self.scaled_data = pd.concat(
            [self.x_scaled, self.y_scaled[self.y_cols]], axis=1)
        self.unscaled_data = pd.concat([self.x, self.y[self.y_cols]], axis=1)

        self.y_md_cols = ['md_'+col for col in self.y_cols]
        self.yerr_md_cols = [col+'_err' for col in self.y_md_cols]

        # Prepare train and test sets
        self.split_train_test()

        data_dict = {
            'id_col': self.id_col, 'x_cols': self.x_cols, 'y_cols': self.y_cols,
            'sel_cols': self.sel_cols, 'n_trains': self.n_trains,
            'n_tests': self.n_tests, 'train_set': self.train_set,
            'test_set': self.test_set, 'x_scaling': self.x_scaling,
            'xscaler': self.xscaler, 'x_scaled': self.x_scaled,
            'y_scaling': self.y_scaling, 'y_mean': self.y_mean,
            'y_std': self.y_std, 'y_max': self.y_max, 'y_min': self.y_min,
            'y_org': self.y, 'y_scaled': self.y_scaled,
            'y_md_cols': self.y_md_cols, 'yerr_md_cols': self.yerr_md_cols,
            'data_size': self.data_size}

        return data_dict

    def load_pred_data(self, predict_params):
        """ 
        Load X data and scale it for predictions 
        """

        data_file = predict_params['data_file']

        data_fp = pd.read_csv(data_file, delimiter=',',
                              header=0, low_memory=False)

    def get_cv_datasets(self, train_set, x_cols, y_cols, train_cv, test_cv):
        """ 
        Cross-validation datasets 

        Given the training set in self.train_set, and two sets of indices, 
        train_inds and test_inds from KFold, the cross-validation datasets
        are returned for training the model
        """

        x_cv_train = np.array(
            train_set.iloc[train_cv][x_cols]).astype(np.float32)
        x_cv_test = np.array(
            train_set.iloc[test_cv][x_cols]).astype(np.float32)
        y_cv_train = np.array(
            train_set.iloc[train_cv][y_cols]).astype(np.float32)
        y_cv_test = np.array(
            train_set.iloc[test_cv][y_cols]).astype(np.float32)

        return x_cv_train, x_cv_test, y_cv_train, y_cv_test
