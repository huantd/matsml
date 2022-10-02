import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math

scaling = 'log'
training_data = pd.read_csv('training.csv')
sel_cols = [col for col in list(training_data.columns) if "selector" in col]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(6, 6))

plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

if scaling == 'log':
    plt.xscale('log')
    plt.yscale('log')
    ymin = min(np.log10(training_data['prop_value'].min()), np.log10(training_data['md_prop_value'].min()))
    ymax = max(np.log10(training_data['prop_value'].max()), np.log10(training_data['md_prop_value'].max()))
    lmin = np.exp(2.30258509299*math.floor(ymin - 0.05* (ymax - ymin)))
    lmax = np.exp(2.30258509299*math.ceil(ymax + 0.05* (ymax - ymin)))
else:
    ymin = min(training_data['prop_value'].min(), training_data['md_prop_value'].min())
    ymax = max(training_data['prop_value'].max(), training_data['md_prop_value'].max())
    lmin = ymin - 0.05*(ymax - ymin)
    lmax = ymax + 0.05*(ymax - ymin)

plt.xlim(lmin, lmax)
plt.ylim(lmin, lmax)

rmse_train = np.sqrt(np.mean((training_data['prop_value'] - training_data['md_prop_value'])**2))
r2_train = r2_score(training_data['prop_value'], training_data['md_prop_value'])

plt.tick_params(axis='x', which='both', bottom=True, top=False,
                labelbottom=True)
plt.tick_params(axis='y', which='both', direction='in')
plt.ylabel("Predicted value", size=12)
plt.xlabel("Reference value", size=12)

for i in range(len(sel_cols)):
    sel_col = sel_cols[i]
    sel_data = training_data[training_data[sel_col] == 1]
    plt.scatter(sel_data['prop_value'], sel_data['md_prop_value'], c = colors[i])

#plt.text(lmin-0.05*(lmax-lmin), lmax+0.0*(lmax-lmin), r'(rmse & $R^2$) = (%.3f & %.3f)' % (rmse_train, r2_train))
plt.plot([lmin, lmax],[lmin, lmax], color = 'red', linewidth = 3.0)

plt.legend(loc="lower right", fontsize=11)

plt.savefig('model_all.pdf')

print('    training, (rmse & R2) = ( %.3f & %.3f )' %(rmse_train, r2_train))
plt.close()

for sel_col in sel_cols:
    sel_data = training_data[training_data[sel_col] == 1]

    plt.figure(figsize=(6, 6))

    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.xscale('log')
    plt.yscale('log')
    #lmin = min(sel_data['prop_value'].min(), sel_data['md_prop_value'].min())
    #lmax = max(sel_data['prop_value'].max(), sel_data['md_prop_value'].max())
    #plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
    #plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
    if scaling == 'log':
        plt.xscale('log')
        plt.yscale('log')
        ymin = min(np.log10(sel_data['prop_value'].min()), np.log10(sel_data['md_prop_value'].min()))
        ymax = max(np.log10(sel_data['prop_value'].max()), np.log10(sel_data['md_prop_value'].max()))
        lmin = np.exp(2.30258509299*math.floor(ymin - 0.05* (ymax - ymin)))
        lmax = np.exp(2.30258509299*math.ceil(ymax + 0.05* (ymax - ymin)))
    else:
        ymin = min(sel_data['prop_value'].min(), sel_data['md_prop_value'].min())
        ymax = max(sel_data['prop_value'].max(), sel_data['md_prop_value'].max())
        lmin = ymin - 0.05*(ymax - ymin)
        lmax = ymax + 0.05*(ymax - ymin)

    plt.xlim(lmin, lmax)
    plt.ylim(lmin, lmax)

    rmse_train = np.sqrt(np.mean((sel_data['prop_value'] - sel_data['md_prop_value'])**2))
    r2_train = r2_score(sel_data['prop_value'], sel_data['md_prop_value'])

    plt.tick_params(axis='x', which='both', bottom=True, top=False,
                    labelbottom=True)
    plt.tick_params(axis='y', which='both', direction='in')
    plt.ylabel("Predicted value", size=12)
    plt.xlabel("Reference value", size=12)
    plt.scatter(sel_data['prop_value'], sel_data['md_prop_value'], color='tab:blue', 
            marker='s', alpha=0.95, label=r'training, (rmse & $R^2$) = (%.3f & %.3f)'
            % (rmse_train, r2_train))

    plt.plot([lmin-0.05*(lmax-lmin), lmax+0.05*(lmax-lmin)],[lmin-0.05*(lmax-lmin), lmax+0.05*(lmax-lmin)], color = 'red', linewidth = 3.0)

    plt.legend(loc="lower right", fontsize=11)
    plt.savefig('model_'+str(sel_col)+'.pdf')

    print('    training, (rmse & R2) = ( %.3f & %.3f )' %(rmse_train, r2_train))


