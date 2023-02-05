# By Huan Tran (huantd@gmail.com), 2021
#
import atexit
import pandas as pd
import numpy as np
import ast


@atexit.register
def goodbye():
    print('  ')
    print('  *****')
    print('  matsML job completed')


def get_key(key, dic, val):
    """ get key value from a dic with a default if not found """

    return dic[key] if key in dic else val


class AtomicStructure:
    """ Atomic structure related I/O. More to come. """

    def __init__(self):
        self.status = 'Init'

    def read_xyz(self, filename):
        """
        Functionality: 
            Read a xyz file and return all the information obtained
        Input:     
            filename: string, name of the file to be read
        Returns:
            nat:      integer, number of atoms
            nspecs:   integer, number of species
            specs:    list of species
            xyz_df:   dataframe, species and xyz coords
        """
        xyz = open(str(filename), "r+")
        Lines = xyz.readlines()
        nlines = len(Lines)
        nat = int(Lines[0].strip('\n').strip('\t').strip(' '))
        columns = ['specs', 'x', 'y', 'z']
        xyz_df = pd.DataFrame(columns=columns)
        specs = []
        nspecs = 0
        for i in range(2, nat+2, 1):
            spec = Lines[i].split()[0]
            if (not any(sp == spec for sp in specs)):
                specs.append(spec)
                nspecs += 1
            x = Lines[i].split()[1]
            y = Lines[i].split()[2]
            z = Lines[i].split()[3]
            xyz_df.loc[len(xyz_df)] = [spec, x, y, z]

        return nat, nspecs, specs, xyz_df

    def save_poscar(self, struct_dic, filename):
        """ 
        Functionality
            Save the atomic structure passed to a poscar-format file

        Input
            struct_dic: a dictionary containing "a", "b", "c", "alpha", "beta", "gamma", "nat", 
                        "ntypat", "species", "coordinates", "ref"

            "a", "b", "c":              real, lattice parameters
            "alpha", "beta", "gamma":   real, lattice angles
            "nat":                      integer, number of atoms
            "ntypat":                   integer, number of atom types (species)
            "species":                  list of nat names of nat atoms
            "coordinates":              list of nat lists, each of which is [x, y, z] in reduced units
            "ref":                      string, reference (origin) of the atomic structure

        Output 
            filename: string, name of the file to be read
        """

        # Conversion rate
        conv = np.pi/180

        # Writing format of 3 columns
        cols3 = "{:15.9f} {:15.9f} {:15.9f}"

        # Lattice parameters
        a = struct_dic["a"]
        b = struct_dic["b"]
        c = struct_dic["c"]

        # Lattice angles
        alpha = conv * float(struct_dic["alpha"])
        beta = conv * float(struct_dic["beta"])
        gamma = conv * float(struct_dic["gamma"])

        # Number of atoms
        nat = int(struct_dic["nat"])

        # Number of atom types
        ntypat = int(struct_dic["ntypat"])

        # List of atom's species
        species = [ast.literal_eval(struct_dic['species'])[
            i].replace(' ', '') for i in range(nat)]

        # Coordinates of atoms in relative unit (wrt lattice)
        coordinates = ast.literal_eval(struct_dic['coordinates'])

        # references of the structure
        ref = struct_dic["ref"]

        # Further standardize the data
        spec_df = pd.DataFrame(species, columns=['species'])
        xred_df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
        coordinates_df = pd.concat(
            [spec_df, xred_df], axis=1).sort_values(by='species')
        coordinates_df.reset_index(inplace=True)

        # List of type of atom
        typat = sorted(list(set(list(coordinates_df['species']))))
        netypat = [str(species.count(spec)).rjust(6) for spec in typat]
        typat_formated = [spec.rjust(6) for spec in typat]

        with open(filename, 'w') as out_file:
            out_file.write(" references: " + ref + '\n')
            out_file.write("{:7.3f}".format(1) + '\n')

            # Three lattice parameters, a bit math needed & done for c
            # a
            out_file.write(cols3.format(a, 0, 0) + '\n')
            # b
            out_file.write(cols3.format(
                b*np.cos(gamma), b*np.sin(gamma), 0) + '\n')
            # c
            cx = c*np.cos(beta)
            cy = c/np.sin(gamma)*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))
            cz = np.sqrt(c*c-cx*cx-cy*cy)
            out_file.write(cols3.format(cx, cy, cz) + '\n')

            # Species
            out_file.write(' '.join(typat_formated) + '\n')
            out_file.write(' '.join(netypat) + '\n')
            out_file.write('Direct' + '\n')

            # Now the reduced coordinates
            for i in range(nat):
                x = coordinates_df.at[i, 'x']
                y = coordinates_df.at[i, 'y']
                z = coordinates_df.at[i, 'z']
                out_file.write(cols3.format(x, y, z) + '\n')


def progress_bar(i_loop, loop_length, action):
    """ Progress bar for some slow works """

    import sys

    toolbar_width = 50
    toolbar_step = loop_length/toolbar_width
    if action == 'update':
        sys.stdout.write("    [%-50s] %d%%" % ('='*min(int(i_loop/toolbar_step) +
                                                       1, 100), int(100/toolbar_width*i_loop/toolbar_step+1)))
        sys.stdout.flush()
    elif action == 'finish':
        sys.stdout.write('\n')


def plot_det_preds(y_cols, y_md_cols, n_tests, log_scaling, pdf_output):
    """ 
    Plot results of the models trained and saved in training.csv and 
        test.csv.

    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    print('  Plot results in "training.csv" & "test.csv"')

    train_df = pd.read_csv('training.csv')
    n_trains = len(train_df)
    if n_tests > 0:
        test_df = pd.read_csv('test.csv')
        n_tests = len(test_df)

    for y_col, y_md_col in zip(y_cols, y_md_cols):
        plt.figure(figsize=(6, 6))

        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        if n_tests > 0:
            lmin = min(test_df[y_col].min(), train_df[y_col].min(),
                       test_df[y_md_col].min(), train_df[y_md_col].min())
            lmax = max(test_df[y_col].max(), train_df[y_col].max(),
                       test_df[y_md_col].max(), train_df[y_md_col].max())
        else:
            lmin = min(train_df[y_col].min(), train_df[y_md_col].min())
            lmax = max(train_df[y_col].max(), train_df[y_md_col].max())

        if log_scaling:
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
            plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))

        rmse_train = np.sqrt(np.mean((train_df[y_col]-train_df[y_md_col])**2))
        r2_train = r2_score(train_df[y_col], train_df[y_md_col])
        if n_tests > 0:
            rmse_test = np.sqrt(np.mean((test_df[y_col]-test_df[y_md_col])**2))
            r2_test = r2_score(test_df[y_col], test_df[y_md_col])

        plt.tick_params(axis='x', which='both', bottom=True, top=False,
                        labelbottom=True)
        plt.tick_params(axis='y', which='both', direction='in')
        plt.ylabel("Predicted value", size=12)
        plt.xlabel("Reference value", size=12)
        plt.scatter(train_df[y_col], train_df[y_md_col], color='tab:red',
                    marker='s', alpha=0.95,
                    label=r'training, (rmse & $R^2$) = (%.3f & %.3f)'
                    % (rmse_train, r2_train))

        print('    training, (rmse & R2) = ( %.3f & %.3f )' %
              (rmse_train, r2_train))
        if n_tests > 0:
            plt.scatter(test_df[y_col], test_df[y_md_col], color='tab:blue',
                        marker='o', alpha=0.6,
                        label=r'test, (rmse & $R^2$) = (%.3f & %.3f)'
                        % (rmse_test, r2_test))
            print('    test, (rmse & R2) = ( %.3f & %.3f )' %
                  (rmse_test, r2_test))
        plt.legend(loc="lower right", fontsize=11)

        if pdf_output:
            plt.savefig('model_'+str(y_col)+'.pdf')
            print('    model_'+str(y_col)+'.pdf saved')
        else:
            print('    showing '+str(y_col))
            plt.show()
        plt.close()


def get_struct_params(data_loc, struct_df):
    from ase import io
    import os

    specs = []
    nats = []
    for k, v in struct_df['file_name'].items():
        struct = io.read(os.path.join(data_loc, str(v)))
        specs = specs + list(set(struct.get_chemical_symbols()))
        nats = nats + [len(struct)]
    specs = list(set(specs))

    return max(nats), len(specs), specs


def plot_prob_preds(y_cols, y_md_cols, yerr_md_cols, n_tests, pdf_output):
    """ 
    Plot results of the models trained and saved in training.csv and 
        test.csv.

    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    print('')
    print('  Plot results in "training.csv" & "test.csv"')

    train_df = pd.read_csv('training.csv')
    n_trains = len(train_df)
    if n_tests > 0:
        test_df = pd.read_csv('test.csv')
        n_tests = len(test_df)

    # First, plot the parity plots
    for y_col, y_md_col, yerr_md_col in zip(y_cols, y_md_cols, yerr_md_cols):
        plt.figure(figsize=(6, 6))

        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        if n_tests > 0:
            lmin = min(test_df[y_col].min(), train_df[y_col].min(),
                       test_df[y_md_col].min(), train_df[y_md_col].min())
            lmax = max(test_df[y_col].max(), train_df[y_col].max(),
                       test_df[y_md_col].max(), train_df[y_md_col].max())
        else:
            lmin = min(train_df[y_col].min(), train_df[y_md_col].min())
            lmax = max(train_df[y_col].max(), train_df[y_md_col].max())

        plt.xlim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))
        plt.ylim(lmin-0.1*(lmax-lmin), lmax+0.1*(lmax-lmin))

        rmse_train = np.sqrt(np.mean((train_df[y_col]-train_df[y_md_col])**2))
        r2_train = r2_score(train_df[y_col], train_df[y_md_col])
        if n_tests > 0:
            rmse_test = np.sqrt(np.mean((test_df[y_col]-test_df[y_md_col])**2))
            r2_test = r2_score(test_df[y_col], test_df[y_md_col])

        plt.tick_params(axis='x', which='both', bottom=True, top=False,
                        labelbottom=True)
        plt.tick_params(axis='y', which='both', direction='in')
        plt.ylabel("Predicted value", size=12)
        plt.xlabel("Reference value", size=12)
        plt.errorbar(train_df[y_col], train_df[y_md_col], yerr=train_df[yerr_md_col], color='tab:red', fmt='s', alpha=0.95,
                     label=r'training, (rmse & $R^2$) = ( %.3f & %.3f )'
                     % (rmse_train, r2_train))
        if n_tests > 0:
            plt.errorbar(test_df[y_col], test_df[y_md_col], yerr=test_df[yerr_md_col], color='tab:blue', fmt='o', alpha=0.6,
                         label=r'test, (rmse & $R^2$) = ( %.3f & %.3f )'
                         % (rmse_test, r2_test))
        plt.legend(loc="lower right", fontsize=11)
        if pdf_output:
            plt.savefig('model_'+str(y_col)+'.pdf')
            print('    model_'+str(y_col)+'.pdf saved')
        else:
            print('    showing '+str(y_col))
            plt.show()
        plt.close()
