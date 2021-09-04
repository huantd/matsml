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
    """ Progress bar """
    import sys

    toolbar_width=50
    toolbar_step=loop_length/toolbar_width
    if action=='update':
        sys.stdout.write("    [%-50s] %d%%"%('='*int(i_loop/toolbar_step),
            int(100/toolbar_width*i_loop/toolbar_step+1)))
        sys.stdout.flush()
    elif action=='finish':
        sys.stdout.write('\n')

def plot_results():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    print ('')
    print ('  Plot results in "training.csv" & "test.csv"')
    test_df=pd.read_csv('test.csv')
    train_df=pd.read_csv('training.csv')
    n_trains=len(train_df)
    n_tests=len(test_df)
    plt.figure(figsize=(6, 6))
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)

    #plt.xlim([-120,0])
    #plt.xlim([-120,0])

    rmse_train=np.sqrt(np.mean((train_df['target']-train_df['md_target'])**2))
    r2_train=r2_score(train_df['target'],train_df['md_target'])
    rmse_test=np.sqrt(np.mean((test_df['target']-test_df['md_target'])**2))
    r2_test=r2_score(test_df['target'],test_df['md_target'])
    plt.text(-110,-35,'n_trains: %s points\nn_tests: %s points\ntraining rmse: %.3f (eV)\ntest rmse: %.3f (eV)\ntraining r2: %.3f (eV)\ntest r2: %.3f (eV)'
        %(n_trains,n_tests,rmse_train,rmse_test,r2_train,r2_test),size=11)

    plt.tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=True)
    plt.tick_params(axis='y',which='both',direction='in')
    plt.ylabel("Predicted value", size=14)
    plt.xlabel("Reference value", size = 14)
    plt.scatter(train_df['target'],train_df['md_target'],color='tab:red',alpha = 0.5,label='training set')
    plt.scatter(test_df['target'],test_df['md_target'],color='tab:blue',alpha = 0.5,label='testset')
    plt.legend(loc="lower right", fontsize = 13)
    plt.show()
