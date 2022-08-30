import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


training_data = pd.read_csv('training.csv')
sel_cols = [col for col in list(training_data.columns) if "selector" in col]
for sel_col in sel_cols:
    sel_data = training_data[training_data[sel_col] == 1]

    plt.figure(figsize=(6, 6))

    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)

    lmin = min(sel_data['prop_value'].min(), sel_data['md_prop_value'].min())
    lmax = max(sel_data['prop_value'].max(), sel_data['md_prop_value'].max())

    plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
    plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))

    rmse_train = np.sqrt(np.mean((sel_data['prop_value'] - sel_data['md_prop_value'])**2))
    r2_train = r2_score(sel_data['prop_value'], sel_data['md_prop_value'])

    plt.tick_params(axis='x', which='both', bottom=True, top=False,
                    labelbottom=True)
    plt.tick_params(axis='y', which='both', direction='in')
    plt.ylabel("Predicted value", size=12)
    plt.xlabel("Reference value", size=12)
    plt.scatter(sel_data['prop_value'], sel_data['md_prop_value'], color='tab:red', 
            marker='s', alpha=0.95, label=r'training, (rmse & $R^2$) = (%.3f & %.3f)'
            % (rmse_train, r2_train))

    plt.legend(loc="lower right", fontsize=11)
    plt.savefig('model_'+str(sel_col)+'.pdf')

    print('    training, (rmse & R2) = ( %.3f & %.3f )' %(rmse_train, r2_train))


