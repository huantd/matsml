# By Huan Tran (huantd@gmail.com), 2021
# 
import atexit
import pandas as pd
import numpy as np

@atexit.register
def goodbye():
    print ('  ')
    print ('  *****')
    print ('  matsML job completed')


class AtomicStructure:
    """ Atomic structure related I/O. More to come. """
    def __init__(self):
        self.status='Init'

    def read_xyz(self,filename):
        """
        Functionality: 
            Read the xyz file and return all the information obtained
        Input:     
            filename: string, name of the file to be read
        Returns:
            nat:      integer, number of atoms
            nspecs:   integer, number of species
            specs:    list of species
            xyz_df:   dataframe, species and xyz coords
        """
        xyz=open(str(filename),"r+")  
        Lines=xyz.readlines()
        nlines=len(Lines)
        nat=int(Lines[0].strip('\n').strip('\t').strip(' '))
        columns=['specs','x','y','z']
        xyz_df=pd.DataFrame(columns=columns)
        specs=[]
        nspecs=0
        for i in range(2,nat+2,1):
            spec=Lines[i].split()[0]
            if (not any (sp==spec for sp in specs)):
                specs.append(spec)
                nspecs+=1
            x=Lines[i].split()[1]
            y=Lines[i].split()[2]
            z=Lines[i].split()[3]
            xyz_df.loc[len(xyz_df)]=[spec,x,y,z]
        
        return nat,nspecs,specs,xyz_df


def progress_bar(i_loop,loop_length,action):
    """ Progress bar for some slow works """

    import sys

    toolbar_width=50
    toolbar_step=loop_length/toolbar_width
    if action=='update':
        sys.stdout.write("    [%-50s] %d%%"%('='*min(int(i_loop/toolbar_step)+\
            1,100),int(100/toolbar_width*i_loop/toolbar_step+1)))
        sys.stdout.flush()
    elif action=='finish':
        sys.stdout.write('\n')


def plot_det_preds(y_cols,y_md_cols,pdf_output):
    """ 
    Plot results of the models trained and saved in training.csv and 
        test.csv.

    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    print ('')
    print ('  Plot results in "training.csv" & "test.csv"')
    test_df=pd.read_csv('test.csv')
    train_df=pd.read_csv('training.csv')
    n_trains=len(train_df)
    n_tests=len(test_df)

    for y_col,y_md_col in zip(y_cols,y_md_cols):
        plt.figure(figsize=(6,6))
        
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        lmin=min(test_df[y_col].min(),train_df[y_col].min(),
            test_df[y_md_col].min(),train_df[y_md_col].min())
        lmax=max(test_df[y_col].max(),train_df[y_col].max(),
            test_df[y_md_col].max(),train_df[y_md_col].max())
        plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
        plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))


        rmse_train=np.sqrt(np.mean((train_df[y_col]-train_df[y_md_col])**2))
        r2_train=r2_score(train_df[y_col],train_df[y_md_col])
        rmse_test=np.sqrt(np.mean((test_df[y_col]-test_df[y_md_col])**2))
        r2_test=r2_score(test_df[y_col],test_df[y_md_col])
        
        plt.tick_params(axis='x',which='both',bottom=True,top=False,
            labelbottom=True)
        plt.tick_params(axis='y',which='both',direction='in')
        plt.ylabel("Predicted value", size=12)
        plt.xlabel("Reference value", size=12)
        plt.scatter(train_df[y_col],train_df[y_md_col],color='tab:red',
            marker='s',alpha=0.95,
            label=r'training, (rmse & $R^2$) = (%.3f & %.3f)'\
            %(rmse_train,r2_train))
        plt.scatter(test_df[y_col],test_df[y_md_col],color='tab:blue',
            marker='o',alpha=0.6,
            label=r'test, (rmse & $R^2$) = (%.3f & %.3f)'\
            %(rmse_test,r2_test))
        plt.legend(loc="lower right",fontsize = 11)
        if pdf_output:
            plt.savefig('model_'+str(y_col)+'.pdf')
            print ('    model_'+str(y_col)+'.pdf saved')
        else:
            print ('    showing '+str(y_col))
            plt.show()
        plt.close()


def plot_prob_preds(y_cols,y_md_cols,yerr_md_cols,pdf_output):
    """ 
    Plot results of the models trained and saved in training.csv and 
        test.csv.

    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    print ('')
    print ('  Plot results in "training.csv" & "test.csv"')
    test_df=pd.read_csv('test.csv')
    train_df=pd.read_csv('training.csv')

    print (train_df)

    n_trains=len(train_df)
    n_tests=len(test_df)

    for y_col,y_md_col,yerr_md_col in zip(y_cols,y_md_cols,yerr_md_cols):
        plt.figure(figsize=(6,6))
        
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        lmin=min(test_df[y_col].min(),train_df[y_col].min(),
            test_df[y_md_col].min(),train_df[y_md_col].min())
        lmax=max(test_df[y_col].max(),train_df[y_col].max(),
            test_df[y_md_col].max(),train_df[y_md_col].max())
        plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
        plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))


        rmse_train=np.sqrt(np.mean((train_df[y_col]-train_df[y_md_col])**2))
        r2_train=r2_score(train_df[y_col],train_df[y_md_col])
        rmse_test=np.sqrt(np.mean((test_df[y_col]-test_df[y_md_col])**2))
        r2_test=r2_score(test_df[y_col],test_df[y_md_col])
        
        plt.tick_params(axis='x',which='both',bottom=True,top=False,
            labelbottom=True)
        plt.tick_params(axis='y',which='both',direction='in')
        plt.ylabel("Predicted value", size=12)
        plt.xlabel("Reference value", size=12)
        plt.errorbar(train_df[y_col],train_df[y_md_col],yerr=\
            train_df[yerr_md_col],color='tab:red',fmt='s',alpha=0.95,
            label=r'training, (rmse & $R^2$) = (%.3f & %.3f)'\
            %(rmse_train,r2_train))
        plt.errorbar(test_df[y_col],test_df[y_md_col],yerr=\
            test_df[yerr_md_col],color='tab:blue',fmt='o',alpha=0.6,
            label=r'test, (rmse & $R^2$) = (%.3f & %.3f)'\
            %(rmse_test,r2_test))
        plt.legend(loc="lower right",fontsize = 11)
        if pdf_output:
            plt.savefig('model_'+str(y_col)+'.pdf')
            print ('    model_'+str(y_col)+'.pdf saved')
        else:
            print ('    showing '+str(y_col))
            plt.show()
        plt.close()
