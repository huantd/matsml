# By Huan Tran (huantd@gmail.com), 2021

import atexit
import pandas as pd
import numpy as np
import json
import sys



@atexit.register
def goodbye():
    """Print a message when the program exits."""
    print('\n  *****\n  matsML job completed\n')


def get_key(key, dic, default_value):
    """
    Retrieve the value associated with a key in a dictionary, 
    returning a default value if the key is not found.

    Parameters:
    key (str): The key to look up in the dictionary.
    dic (dict): The dictionary to search in.
    default_value: The value to return if the key is not found.

    Returns:
    The value associated with the key, or the default value if the key is not found.
    """
    return dic.get(key, default_value)


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
        print('    Read an XYZ atomic structure from ' + str(filename))

        xyz = open(str(filename), "r+")
        Lines = xyz.readlines()
        nlines = len(Lines)
        nat = int(Lines[0].strip('\n').strip('\t').strip(' '))
        columns = ['specs', 'x', 'y', 'z']
        xyz_df = pd.DataFrame(columns=columns)
        specs = []
        nspecs = 0
        for i in range(2, nat + 2, 1):
            spec = Lines[i].split()[0]
            if (not any(sp == spec for sp in specs)):
                specs.append(spec)
                nspecs += 1
            x = Lines[i].split()[1]
            y = Lines[i].split()[2]
            z = Lines[i].split()[3]
            xyz_df.loc[len(xyz_df)] = [spec, x, y, z]

        # struct_dict = {'nat': nat}
        return nat, nspecs, specs, xyz_df

    def read_poscar(self, filename):
        """ 
        Functionality
            Read a poscar and return a struct_dic.

        Input
            filename: string, name of the file to be read

        Output 
            struct_dic: a dictionary containing "a", "b", "c", "alpha", "beta", "gamma", "nat", 
                        "ntypat", "species", "xred", "xcart", "ucvol", "note". 

            "a", "b", "c":              real, lattice parameters
            "alpha", "beta", "gamma":   real, lattice angles
            "nat":                      integer, number of atoms
            "ntypat":                   integer, number of atom types (species)
            "ucvol":                    unit cell volume
            "species":                  list of nat names of nat atoms
            "xred":                     list of nat lists, each of which is [x, y, z] in reduced units
            "xcart":                    list of nat lists, each of which is [x, y, z] in cartesian
            "note":                     string, note, to be written in the first line 

        NOTE: POSCAR format that has additional lines such as "Selective dynamics" current not supported

        """

        print('    Read a POSCAR atomic structure from ' + str(filename))

        # Conversion rate
        conv = 180/np.pi

        poscar = open(str(filename), "r+")
        Lines = poscar.readlines()
        nlines = len(Lines)

        # First line is comment in POSCAR format
        line1 = Lines[1].strip('\n').strip('\t').strip(' ')
        note = 'POSCAR' if len(line1) == 0 else line1

        # Scale of the cell in the second line
        scale = float(Lines[1].strip('\n').strip('\t').strip(' '))

        # Next 3 lines for a, b, and c vectors
        avec = [scale*float(el) for el in Lines[2].split()]
        bvec = [scale*float(el) for el in Lines[3].split()]
        cvec = [scale*float(el) for el in Lines[4].split()]

        # Length of 3 lattice parameters
        a = np.sqrt(dot_prod(avec, avec))
        b = np.sqrt(dot_prod(bvec, bvec))
        c = np.sqrt(dot_prod(cvec, cvec))

        # Unit cell volume
        ucvol = dot_prod(cross_prod(avec, bvec), cvec)

        # Three angles, in degrees
        alpha = np.arccos(dot_prod(bvec, cvec)/(b*c))*conv
        beta = np.arccos(dot_prod(avec, cvec)/(a*c))*conv
        gamma = np.arccos(dot_prod(bvec, avec)/(b*a))*conv

        # List of species
        list_species = Lines[5].split()

        # Number of atom types
        ntypat = len(list_species)

        # Number of each type of atoms
        netypat = [int(el) for el in Lines[6].split()]

        # Species in details
        species_tmp = []
        for ityp in range(len(netypat)):
            species_tmp = species_tmp + [list_species[ityp]
                                         for iat in range(netypat[ityp])]
        species = '%s' % json.dumps(species_tmp)

        # Number of atoms
        nat = sum(netypat)

        # Direct or Cartesian, compute and return xred and xcart
        poscar_mode_line = str(Lines[7].strip('\n').strip('\t').strip(' '))

        if (poscar_mode_line.startswith('D')) or (poscar_mode_line.startswith('d')):
            poscar_mode = 'Direct'
            rprim = np.transpose(np.array([avec, bvec, cvec]))
            xred = []
            for iat in range(8, nat+8, 1):
                xred.append([float(Lines[iat].split()[i]) for i in range(3)])

            xcart = np.transpose(np.matmul(rprim, np.transpose(xred)))

        elif (poscar_mode_line.startswith('C')) or (poscar_mode_line.startswith('c')):
            poscar_mode = 'Cartesian'
            rprim_inv = np.linalg.inv(
                np.transpose(np.array([avec, bvec, cvec])))
            xcart = []
            for iat in range(8, nat+8, 1):
                xcart.append([float(Lines[iat].split()[i]) for i in range(3)])
            xred = np.transpose(
                np.matmul(rprim_inv, np.transpose(np.array(xcart))))

        # Compile two lists
        xred_list = '%s' % json.dumps(
            [[np.array(xred)[iat, i] for i in range(3)] for iat in range(nat)])
        xcart_list = '%s' % json.dumps(
            [[np.array(xcart)[iat, i] for i in range(3)] for iat in range(nat)])

        # output as a dictionary
        struct_dic = {'nat': nat, 'ntypat': ntypat, 'a': a, 'b': b, 'c': c,
                      'alpha': alpha, 'beta': beta, 'gamma': gamma, 'species': species,
                      'xred': xred_list, 'xcart': xcart_list, 'ucvol': ucvol, 'note': note}

        return struct_dic

    def write_poscar(self, struct_dic, filename):
        """ 
        Functionality
            Write the crystalline atomic structure passed to a poscar-format file. In the output,
            vector a of the box is aligned along the x axis, vector b is on the xy plane, and vector 
            c is free, i.e., the lattice parameters form a lower triangular square (3x3) matrix.  

        Input
            struct_dic: a dictionary containing "a", "b", "c", "alpha", "beta", "gamma", "nat", 
                        "ntypat", "species", "xred", "xcart", "ucvol", "note". 

            "a", "b", "c":              real, lattice parameters
            "alpha", "beta", "gamma":   real, lattice angles
            "nat":                      integer, number of atoms
            "ntypat":                   integer, number of atom types (species)
            "ucvol":                    unit cell volume
            "species":                  list of nat names of nat atoms
            "xred":                     list of nat lists, each of which is [x, y, z] in reduced units
            "xcart":                    list of nat lists, each of which is [x, y, z] in cartesian
            "note":                     string, note, to be written in the first line 

            filename: string, name of the file to be writen
        """

        print('    Write a POSCAR atomic structure to ' + str(filename))

        # Conversion rate
        conv = np.pi/180

        # Writing format of 3 columns
        cols3 = "{:15.9f} {:15.9f} {:15.9f}"

        # Lattice parameters
        a = struct_dic['a']
        b = struct_dic['b']
        c = struct_dic['c']

        # Lattice angles
        alpha = conv * float(struct_dic['alpha'])
        beta = conv * float(struct_dic['beta'])
        gamma = conv * float(struct_dic['gamma'])

        # Number of atoms
        nat = int(struct_dic['nat'])

        # Number of atom types
        ntypat = int(struct_dic['ntypat'])

        # List of atom's species
        species = [json.loads(struct_dic['species'])[
            i].replace(' ', '') for i in range(nat)]

        # Coordinates of atoms in relative unit (wrt lattice)
        xred = json.loads(struct_dic['xred'])

        # note on the structure
        note = struct_dic['note']

        # Further standardize the data
        spec_df = pd.DataFrame(species, columns=['species'])
        xred_df = pd.DataFrame(xred, columns=['x', 'y', 'z'])
        sorted_xred_df = pd.concat(
            [spec_df, xred_df], axis=1).sort_values(by='species')
        sorted_xred_df.reset_index(inplace=True)

        # List of type of atom
        typat = sorted(list(set(list(sorted_xred_df['species']))))
        netypat = [str(species.count(spec)).rjust(6) for spec in typat]
        typat_formated = [spec.rjust(6) for spec in typat]

        with open(filename, 'w') as out_file:
            out_file.write(note + '\n')
            out_file.write("{:7.3f}".format(1) + '\n')

            # Three lattice parameters as a lower triangular matrix, a bit math
            # needed for c
            out_file.write(cols3.format(a, 0, 0) + '\n')
            out_file.write(cols3.format(
                b*np.cos(gamma), b*np.sin(gamma), 0) + '\n')
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
                out_file.write(cols3.format(
                    float(sorted_xred_df.at[i, 'x']), float(sorted_xred_df.at[i, 'y']), float(sorted_xred_df.at[i, 'z'])) + '\n')


def progress_bar(current_iteration, total_iterations, action):
    """
    Displays a progress bar to indicate the completion of a loop.

    Parameters:
    current_iteration (int): The current loop iteration (0-indexed).
    total_iterations (int): The total number of iterations in the loop.
    action (str): The action to perform. It can be either 'update' (for updating the progress bar)
                  or 'finish' (to print a newline at the end).
    """

    toolbar_width = 50
    step_size = total_iterations / toolbar_width

    if action == 'update':
        sys.stdout.write(
            "    [%-50s] %d%%" % (
                '=' * min(int(current_iteration / step_size) + 1, 100),
                int(100 / toolbar_width * current_iteration / step_size + 1)
            )
        )
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

    print('  ')
    print('  Plot results in "training.csv" & "test.csv"')

    train_df = pd.read_csv('training.csv')
    n_trains = len(train_df)
    if n_tests > 0:
        test_df = pd.read_csv('test.csv')
        n_tests = len(test_df)

    for y_col, y_md_col in zip(y_cols, y_md_cols):
        plt.figure(figsize=(6, 6))

        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)

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
            plt.xlim(lmin - 0.1*(lmax - lmin), lmax + 0.1*(lmax - lmin))
            plt.ylim(lmin - 0.1*(lmax - lmin), lmax + 0.1*(lmax - lmin))

        rmse_train = np.sqrt(
            np.mean((train_df[y_col] - train_df[y_md_col])**2))
        r2_train = r2_score(train_df[y_col], train_df[y_md_col])
        if n_tests > 0:
            rmse_test = np.sqrt(
                np.mean((test_df[y_col] - test_df[y_md_col])**2))
            r2_test = r2_score(test_df[y_col], test_df[y_md_col])

        plt.tick_params(axis='x', which='both', bottom=True,
                        top=False, labelbottom=True)
        plt.tick_params(axis='y', which='both', direction='in')
        plt.ylabel("Predicted value", size=11)
        plt.xlabel("Reference value", size=11)
        plt.plot([lmin - 0.01*(lmax - lmin), lmax + 0.01*(lmax - lmin)], [lmin -
                 0.01*(lmax - lmin), lmax + 0.01*(lmax - lmin)], color='green', linewidth=2.5)
        plt.scatter(train_df[y_col], train_df[y_md_col], color='tab:red', marker='s', alpha=0.95,
                    label=r'training, (rmse & $R^2$) = (%.3f & %.3f)' % (rmse_train, r2_train))

        print('    training, (rmse & R2) = ( %.3f & %.3f )' %
              (rmse_train, r2_train))
        if n_tests > 0:
            plt.scatter(test_df[y_col], test_df[y_md_col], color='tab:blue', marker='o',
                        alpha=0.6, label=r'test, (rmse & $R^2$) = (%.3f & %.3f)' % (rmse_test, r2_test))
            print('    test, (rmse & R2) = ( %.3f & %.3f )' %
                  (rmse_test, r2_test))
        plt.legend(loc="lower right", fontsize=9)

        if pdf_output:
            plt.savefig('model_' + str(y_col) + '.pdf')
            print('    model_'+str(y_col) + '.pdf saved')
        else:
            print('    showing ' + str(y_col))
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

    print('  ')
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


def dot_prod(a, b):
    '''
    Dot product of 2 lists a and b, each has the same dimensionality
    '''
    if (type(a) is list) and (type(b) is list) and (len(a) == len(b)):
        prod = sum([a[i]*b[i] for i in range(len(a))])
    else:
        raise ValueError(
            '      ERROR, dot_prod: something wrong with 2 inputs')

    return prod


def cross_prod(a, b):
    '''
    Cross product of 2 list a and b, each has 3 elements
    '''

    if (type(a) is list) and (type(b) is list) and (len(a) == 3) and (len(b) == 3):
        prod = [(a[1]*b[2] - a[2]*b[1]), (a[2]*b[0] -
                                          a[0]*b[2]), (a[0]*b[1] - a[1]*b[0])]
    else:
        raise ValueError(
            '      ERROR, cross_prod: something wrong with 2 inputs')

    return prod
