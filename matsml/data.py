# Copyright Huan Tran (huantd@gmail.com), 2021
#
# Works with data
#
#
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
import pandas as pd

class ProcessData:

    def __init__(self,data_params):
        self.data_params = data_params

    def get_xy(self):
        print ('  - Reading data ... ')

        data_file = self.data_params['data_file']

        id_col = self.data_params['id_col']
        if len(id_col) > 1:
            raise ValueError('  ERROR: There must be one ID column only')
        else:
            self.id_col = id_col

        y_cols = self.data_params['y_cols']
        self.y_cols = y_cols

        y_dim = len(y_cols)
        self.y_dim = y_dim

        comment_cols = self.data_params['comment_cols']
        self.comment_cols = comment_cols

        fingerprint_data = pd.read_csv(data_file,delimiter=',',header=0,
                low_memory=False)

        self.data_size = len(fingerprint_data)

        # list of columns for x
        x_cols = [col for col in list(fingerprint_data.columns) if col not in 
                (y_cols+id_col+comment_cols)]
        self.x_cols = x_cols

        # list of onehot cols
        onehot_cols = [col for col in x_cols if "onehot" in col] 
        self.onehot_cols = onehot_cols

        # values of onehot cols
        onehot_vals = fingerprint_data[onehot_cols].drop_duplicates() 
        self.onehot_vals = onehot_vals

        # x and y data for learning
        self.y = fingerprint_data[id_col+onehot_cols+y_cols]
        self.x = fingerprint_data[id_col+x_cols]


    def scale_xy(self):
        ''' Scale x and y before learning 
        
            I want to handle onehot and scaling here, so things are a bit 
            complicated. First, I must keep track the onehot values so the 
            inverse scaling can be done. Then, I also wants to keep track
            of the ID so the information of the predictions can be completed.
            Therefore, dont surprise of the following complicated lines.
            I will make it simpler latter. 

            Now, my assumption for the one-hot vectors are
               (1) One-hot columns are named "onehot-*"
               (2) Only one element of the one-hot vector is non-zero 
                   and it must be 1.
        '''

        # Work with x data
        print ('  - Data preprocessing ...')
        x = self.x
        self.x_scaling = self.data_params['x_scaling']
        print ('    Scaling x:', self.x_scaling)
        
        if self.x_scaling == 'log':
            x_scaled = np.array(x.drop(self.id_col,axis=1))
            x[:,:] = np.log(x[:,:])
            xmax = np.amax(np.array(x))
            xmin = np.amin(np.array(x))
            x_scaled[:,:] = -1+2*(x[:,:]-xmin)/(xmax-xmin) 
        elif self.x_scaling == 'normalize':
            x_scaled=preprocessing.normalize(x.drop(self.id_col,axis=1), 
                    norm='l2')
        elif self.x_scaling == 'minmax':
            min_max_scaler=preprocessing.MinMaxScaler()
            x_scaled=min_max_scaler.fit_transform(x.drop(self.id_col,axis=1))
        elif self.x_scaling == 'quantile':
            quant_transf=preprocessing.QuantileTransformer(random_state=0)
            x_scaled=quant_transf.fit_transform(x.drop(self.id_col,axis=1))
        elif self.x_scaling == 'yeo-johnson':
            pt=preprocessing.PowerTransformer(method='yeo-johnson', 
                    standardize=False)
            x_scaled=pt.fit_transform(x.drop(self.id_col,axis=1))
        elif self.x_scaling == 'none':
            x_scaled=x.drop(self.id_col,axis=1)
        
        # Convert nparray back pandas and stack the ID column
        x_scaled_df=pd.DataFrame(x_scaled,columns=self.x_cols)
        x_scaled_df[self.id_col]=x[self.id_col]
        self.x_scaled=x_scaled_df
        #print (x_scaled_df)

        # Work with y data
        y = self.y
        y_cols = self.y_cols
        y_dim = self.y_dim
        onehot_cols = self.onehot_cols
        self.y_scaling = self.data_params['y_scaling']
        print ('    Scaling y:', self.y_scaling)

        # Compute some distribution parameters and do scaling
        if len(onehot_cols) > 0:
            y_mean = pd.DataFrame(columns = ['onehot']+y_cols)
            y_std = pd.DataFrame(columns = ['onehot']+y_cols)
            y_min = pd.DataFrame(columns = ['onehot']+y_cols)
            y_max = pd.DataFrame(columns = ['onehot']+y_cols)
            
            for onehot in onehot_cols:
                sel_y = y[y[onehot] == 1]
            
                this_row = pd.DataFrame([np.mean(np.array(sel_y[col])) for col 
                    in y_cols], columns = y_cols)
                this_row['onehot'] = onehot
                y_mean = y_mean.append(this_row, ignore_index=True)
            
                this_row = pd.DataFrame([np.std(np.array(sel_y[col])) for col 
                    in y_cols], columns = y_cols)
                this_row['onehot'] = onehot
                y_std = y_std.append(this_row, ignore_index=True)
            
                this_row = pd.DataFrame([np.amin(np.array(sel_y[col])) for col 
                    in y_cols], columns = y_cols)
                this_row['onehot'] = onehot
                y_min = y_min.append(this_row, ignore_index=True)
            
                this_row = pd.DataFrame([np.amax(np.array(sel_y[col])) for col 
                    in y_cols], columns = y_cols)
                this_row['onehot'] = onehot
                y_max = y_max.append(this_row, ignore_index=True)
            
            self.y_mean = y_mean
            self.y_std = y_std
            self.y_min = y_min
            self.y_max = y_max

            y_scaled = pd.DataFrame(columns=self.id_col+onehot_cols+y_cols)
            y_scaled[self.id_col] = y[self.id_col] 
            y_scaled[onehot_cols] = y[onehot_cols] 

            for i, j, onehot in ((a,b,c) for a in range(len(y)) for b in 
                    range(y_dim) for c in onehot_cols):
                this_row = y.iloc[i]
                if this_row[onehot].astype(int)==1:
                    ymean=float(y_mean.loc[y_mean['onehot']==onehot][y_cols[j]])
                    ystd=float(y_std.loc[y_std['onehot']==onehot][y_cols[j]])
                    ymin=float(y_min.loc[y_min['onehot']==onehot][y_cols[j]])
                    ymax=float(y_max.loc[y_max['onehot']==onehot][y_cols[j]])
                    if str(self.y_scaling)=='normalize':
                        y_scaled.at[i,y_cols[j]]=(y.at[i,y_cols[j]]-ymean)/ystd
                    elif str(self.y_scaling)=='minmax':
                        y_scaled.at[i,y_cols[j]]=(y.at[i,y_cols[j]]-ymin)/(ymax-ymin)
                    elif str(self.y_scaling)=='none':
                        y_scaled.at[i,y_cols[j]]=y.at[i,y_cols[j]]

            self.y_scaled = y_scaled
            
        elif len(onehot_cols) == 0:
            self.y_mean=pd.DataFrame([np.mean(np.array(y[col])) for col in y_cols], 
                    columns=y_cols)
            self.y_std=pd.DataFrame([np.std(np.array(y[col])) for col in y_cols], 
                    columns=y_cols)
            self.y_min=pd.DataFrame([np.amin(np.array(y[col])) for col in y_cols], 
                    columns=y_cols)
            self.y_max=pd.DataFrame([np.amax(np.array(y[col])) for col in y_cols], 
                    columns=y_cols)

            y_scaled = pd.DataFrame(columns=self.id_col+y_cols)
            for i, j in ((a,b) for a in range(len(y)) for b in range(y_dim)):
                y_scaled[self.id_col] = y[self.id_col] 
                if str(self.y_scaling) == 'normalize':
                    delta_y=y.at[i,y_cols[j]]-self.y_mean.at[0,y_cols[j]]
                    y_scaled.at[i,y_cols[j]]=delta_y/self.y_std.at[0,y_cols[j]]
                elif str(self.y_scaling) == 'minmax':
                    delta_y=(self.y_max.at[0,y_cols[j]]-self.y_min.at[0,y_cols[j]])
                    y_scaled.at[i,y_cols[j]]=(y.at[i,y_cols[j]]-\
                            self.y_min.at[0,y_cols[j]])/delta_y
                elif str(self.y_scaling) == 'none':
                    y_scaled.at[i,y_cols[j]]=y.at[i,y_cols[j]]

            self.y_scaled = y_scaled
        
        self.scaled_data=pd.concat([self.x_scaled,self.y_scaled[y_cols]],axis=1)

    def prepare_train_test_data(self):
        ''' Prepare train and test sets using the sampling method specified'''


        print ('    Prepare train/test sets: ', self.data_params['sampling'])

        self.n_trains = int(self.data_params['n_trains']*self.data_size)
        self.n_tests = self.data_size - self.n_trains
        self.sampling = self.data_params['sampling']

        id_col = self.id_col
        scaled_data = self.scaled_data
        if self.sampling == 'random':
            train_set_ids = \
                    scaled_data.sample(n = self.n_trains)[id_col[0]].tolist()
            test_set_ids = [idx for idx in scaled_data[id_col[0]].tolist() 
                    if idx not in train_set_ids]
        else:
            raise ValueError \
                    ('  ERROR: only random sampling in this version.')
        self.train_set=scaled_data[scaled_data[id_col[0]].isin(train_set_ids)]
        self.test_set=scaled_data[scaled_data[id_col[0]].isin(test_set_ids)]


    def unscale_y(scaled_y_data, scaling_dic,message):
        """ Unscale the y data """

        print ('    Unscaling y:', scaling_dic['y_scaling'])

        id_col = scaling_dic['id_col']
        y_cols = scaling_dic['y_cols']
        onehot_cols = scaling_dic['onehot_cols']

        y_scaling = scaling_dic['y_scaling']
        y_org = scaling_dic['y_org']
        y_mean = scaling_dic['y_mean']
        y_std = scaling_dic['y_std']
        y_min = scaling_dic['y_min']
        y_max = scaling_dic['y_max']

        model_y_cols = ['md_'+y_col for y_col in y_cols]
        y_org = y_org[id_col+onehot_cols+y_cols]

        y_dim = len(y_cols)
        ids_list = list(scaled_y_data[id_col[0]])

        # Starting from scaled_y_data, unscale the data in y_cols, and add 
        # to y_org, then select nonan to get unscaled_y_data
        #
        if len(onehot_cols) > 0:
            for idn, jy, onehot in ((a,b,c) for a in ids_list for b in 
                    range(y_dim) for c in onehot_cols):
                idx0 = np.array(y_org[y_org[id_col[0]]==idn].index)[0]
                idx1 = np.array(scaled_y_data[scaled_y_data[id_col[0]]==\
                        idn].index)[0]

                if scaled_y_data.at[idx1,onehot] > 0.0:
                    if str(y_scaling) == 'minmax':
                        ymax = float(y_max.loc[y_max['onehot']==\
                            onehot][y_cols[jy]])
                        ymin = float(y_min.loc[y_min['onehot']==\
                            onehot][y_cols[jy]])
                        y_org.at[idx0,model_y_cols[jy]]=scaled_y_data.at[idx1,
                                model_y_cols[jy]]*(ymax-ymin)+ymin
                    elif str(y_scaling) == 'normalize':
                        ymean = float(y_mean.loc[y_max['onehot']==\
                            onehot][y_cols[jy]])
                        ystd = float(y_std.loc[y_min['onehot']==\
                            onehot][y_cols[jy]])
                        y_org.at[idx0,model_y_cols[jy]]=scaled_y_data.at[idx1,
                                model_y_cols[jy]]*ystd +ymean
                    elif str(y_scaling) == 'none':
                        y_org.at[idx0,model_y_cols[jy]]=scaled_y_data.at[idx1,
                                model_y_cols[jy]]

            unscaled_y_data = y_org.dropna(subset=model_y_cols)

            # Get RMSE of each prop
            for onehot in onehot_cols:
                sel_y_data = unscaled_y_data.loc[y_org[onehot] == 1]
                for y_col in y_cols:
                    this_rmse = np.sqrt(np.mean((np.array(sel_y_data[y_col])-\
                            np.array(sel_y_data['md_'+y_col]))**2))
                    print ("      rmse",str(message).ljust(12),onehot,
                            str(y_col).ljust(16),round(this_rmse,6))

        elif len(onehot_cols) == 0:
            for idn, jy in ((a,b) for a in ids_list for b in range(y_dim)):
                idx0 = np.array(y_org[y_org[id_col[0]]==idn].index)[0]
                idx1 = np.array(scaled_y_data[scaled_y_data[id_col[0]] == \
                        idn].index)[0]
                if str(y_scaling) == 'minmax':
                    delta_y=(y_max.at[0,y_cols[jy]]-y_min.at[0,y_cols[jy]])
                    y_org.at[idx0,model_y_cols[jy]] = scaled_y_data.at[idx1,
                            model_y_cols[jy]]*delta_y+y_min.at[0,y_cols[jy]]
                elif str(y_scaling) == 'normalize':
                    y_org.at[idx0,model_y_cols[jy]]=(scaled_y_data.at[idx1, 
                            model_y_cols[jy]]*y_std.at[0,y_cols[jy]])+\
                            y_mean.at[0,y_cols[jy]]
                elif str(y_scaling) == 'none':
                    y_org.at[idx0,model_y_cols[jy]]=scaled_y_data.at[idx1, 
                            model_y_cols[jy]]

            unscaled_y_data=y_org.dropna(subset=model_y_cols)

            for y_col in y_cols:
                this_rmse=np.sqrt(np.mean((np.array(unscaled_y_data[y_col])-\
                        np.array(unscaled_y_data['md_'+y_col]))**2))
                print ("       rmse",str(message).ljust(12),str(y_col).ljust(16),
                        round(this_rmse,6))

        return unscaled_y_data

