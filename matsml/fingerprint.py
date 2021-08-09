# Copyright Huan Tran (huantd@gmail.com), 2021

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from sklearn import preprocessing
import pandas as pd
from matsml.data import ProcessData

class Fingerprint:

    def __init__(self,input_data):
        self.input_data = input_data


def amotif_molec(struct_df, struct_format):
    """
    Functionality: 
        Compute the fingerints based on the atomic motif
    Input:     
        struct_df: pandas dataframe, the structure to be fingeprinted
    Returns:
        message: string, summary of the fingerprinting
        madfp_df: dataframe, madfp_df updated with the new fingerprint
    """

    #fp_dim = int(fp_dimension)

    #AtomSpecs = {'Specs': ['C','H','O'],'Znucl': [6,1,8], 'Eref':[-37.7760091174, -0.461816662560, -74.9573978895]}
    #specs_df = pd.DataFrame(AtomSpecs, columns = ['Specs', 'Znucl', 'Eref'])

    #print (struct_df)
    #columns = ['id', 'target']
    #for i in range(fp_dim):
    #    columns.append('gfp_'+str(i).zfill(3))

    afp = pd.DataFrame()

    atfp_list = []
    for k, v in struct_df['file_name'].items():
        print ('  processing file %s ' %v)
        target = np.array(struct_df.loc[struct_df['file_name'] == str(v)]['prop']).astype(float)[0]
        nat, nspecs, specs, xyz_df =  read_xyz(str(v))
        message,atfp_list, afp_this = atfp(atfp_list, nat, nspecs, specs, xyz_df)
        afp = pd.concat([afp,afp_this], axis=0, ignore_index=True)
        afp.at[afp.shape[0]-1,'id'] = str(v)
        prop = struct_df[struct_df['file_name'] == str(v)]['prop']
        afp.at[afp.shape[0]-1,'target'] = float(prop)

    afp = afp.fillna(0.0)

    # Rearrange columns
    first_cols = ['id', 'target']
    last_cols = [col for col in afp.columns if col not in first_cols]
    afp = afp[first_cols+last_cols]

    return afp

def gaussian_coulombmatrix_molec(struct_df, struct_format, fp_dimension):
    """
    Functionality: 
        Compute the fingerints based in the moments of the atomic distance of the structures provided
    Input:     
        struct_df: pandas dataframe, the structure to be fingeprinted
    Returns:
        message: string, summary of the fingerprinting
        madfp_df: dataframe, madfp_df updated with the new fingerprint
    """

    fp_dim = int(fp_dimension)

    AtomSpecs = {'Specs': ['C','H','O'],'Znucl': [6,1,8], 'Eref':[-37.7760091174, -0.461816662560, -74.9573978895]}
    specs_df = pd.DataFrame(AtomSpecs, columns = ['Specs', 'Znucl', 'Eref'])

    columns = ['id', 'target']
    for i in range(fp_dim):
        columns.append('gfp_'+str(i).zfill(3))
    gfp = pd.DataFrame(columns = columns)

    xmin = 0
    xmax = 78
    #print ('>>', xmin, xmax)
    ri = np.linspace(xmin, xmax, fp_dim)
    sigma = 1.5*(xmax-xmin)/fp_dim
    #print ('>>', ri, sigma)

    for k, v in struct_df['file_name'].items():
        print ('  processing file %s ' %v)
        target = np.array(struct_df.loc[struct_df['file_name'] == str(v)]['prop']).astype(float)[0]
        nat, nspecs, specs, xyz_df =  read_xyz(str(v))
        cm = np.zeros((nat, nat))
        
        for iat in range(nat-1):
            xcart = xyz_df.iloc[iat][['x', 'y', 'z']]
            xi = np.array(xcart).astype(float)
            spi = np.array(xyz_df.iloc[iat][['species']]).astype(str)[0]
            zi = np.array(specs_df.loc[specs_df['Specs'] == str(spi)]['Znucl']).astype(float)[0]
            eref = np.array(specs_df.loc[specs_df['Specs'] == str(spi)]['Eref']).astype(float)[0]
            #target = target - eref
            for jat in range(iat, nat, 1):
                spj = np.array(xyz_df.iloc[jat][['species']]).astype(str)[0]
                zj = np.array(specs_df.loc[specs_df['Specs'] == str(spj)]['Znucl']).astype(float)[0]
                xcart = xyz_df.iloc[jat][['x', 'y', 'z']]
                xj = np.array(xcart).astype(float)
                dist = np.linalg.norm(xi-xj)
                if iat == jat:
                    cm[iat][jat] = 0.5 * zi ** 2.4  # Diagonal term described by Potential energy of isolated atom
                else:
                    cm[iat][jat] = zi * zj / dist   # Pair-wise repulsion

        #eigenvals, eigenvecs = LA.eig(cm)
        cm_flatten = cm.flatten()
        #xmin = min(xmin,np.amin(cm_flatten))
        #xmax = max(xmax,np.amax(cm_flatten))
        #print ('>>', np.amin(cm_flatten), np.amax(cm_flatten))
        #print (cm_flatten)

        #gcm = [str(v), str(target/nat*27.211)]
        gcm = [str(v), str(target)]

        for i in range(fp_dim):
            gcm_el = sum(gaussian(cm_flatten[j], sigma, ri[i]) for j in range(len(cm_flatten)))
            gcm.append(gcm_el)
        gfp = gfp.append(pd.DataFrame(np.array(gcm).reshape((1, fp_dim+2)), columns = columns))

    return gfp

def read_xyz(filename):
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
    import pandas as pd

    xyz = open(str(filename),"r+")  
    Lines = xyz.readlines()
    nlines = len(Lines)
    nat = int(Lines[0].strip('\n').strip('\t').strip(' '))
    columns = ['species', 'x', 'y', 'z']
    xyz_df = pd.DataFrame(columns = columns)
    specs = []
    nspecs = 0
    for i in range(2, nat+2, 1):
        spec = Lines[i].split()[0]
        if (not any (sp == spec for sp in specs)):
            specs.append(spec)
            nspecs += 1
        x = Lines[i].split()[1]
        y = Lines[i].split()[2]
        z = Lines[i].split()[3]
        xyz_df.loc[len(xyz_df)] = [spec, x, y, z]

    return nat, nspecs, specs, xyz_df

def gaussian(x, sigma, r):
    return 1./(math.sqrt(sigma**math.pi))*np.exp(-sigma*np.power((x - r), 2.))

def atfp(atfp_list, nat, nspecs, specs, xyz_df):
    """
    Functionality:
        Compute the atomic fingerint the obtained polymer POSCAR
    Input:
        str_poscar: vasp poscar, the structure to be fingeprinted
        atfp_df: dataframe, emty pd dataframe with columns obtained from the initial data (fingerprint)
    Returns:
        message: string, summary of the fingerprinting
        atfp_df: dataframe, atfp_df updated with the new fingerprint
    """

    #print ('          - fingerprinting ... ', end='')
    bond_def = pd.read_csv('bonds.csv')
    #atfp_list = list(atfp_df.columns)

    species = list(xyz_df['species'])

    coord_num = [] # list of coordination number of all the atoms
    neighbors = [] # neightbor list, nat rows for nat atoms, in each row all the neighbors are listed

    # get the neighbor list and coordination numbers
    for iat in range(nat):
        cn = 0     # coordination number of a specific atom
        nb_loc = []
        for jat in range(nat):
            if iat != jat:
                #d = str_current.get_distance(iat,jat)
                xyz_iat = np.array([float(xyz_df.at[iat,'x']), float(xyz_df.at[iat,'y']), float(xyz_df.at[iat,'z'])])
                xyz_jat = np.array([float(xyz_df.at[jat,'x']), float(xyz_df.at[jat,'y']), float(xyz_df.at[jat,'z'])])
                d = np.linalg.norm(xyz_iat-xyz_jat)
                selected_row = bond_def.loc[ ((bond_def['end1'] == str(species[iat])) & (bond_def['end2'] == str(species[jat]))) |
                        ((bond_def['end1'] == str(species[jat])) & (bond_def['end2'] == str(species[iat])))]
                if selected_row.shape[0] > 0 :
                    if (d >= float(selected_row['lmin'])) and (d <= float(selected_row['lmax'])) :
                        cn = cn + 1
                        nb_loc.append(jat)
        # Update neighbor list and coordination numbers
        coord_num.append(cn)
        neighbors.append(nb_loc)

    # atfp_df.loc[len(atfp_df)] = 0
    atfp_df = pd.DataFrame(0, index=range(1), columns=atfp_list) 

    # look at the neighbor list and compute fingerprint
    message = 'Good: all fingerprints found'
    for iat in range(nat):
        iatn = str(species[iat]) + str(coord_num[iat])
        nb_loc = neighbors[iat]
        for i in range(len(nb_loc)):
            for j in range(len(nb_loc)):
                if i != j:
                    nb1n = str(species[nb_loc[i]]) + str(coord_num[nb_loc[i]])
                    nb2n = str(species[nb_loc[j]]) + str(coord_num[nb_loc[j]])
                    triple1 = nb1n + iatn + nb2n
                    triple2 = nb2n + iatn + nb1n

                    #print (atfp_list, triple1, triple2)

                    if (any (fp == triple1 for fp in atfp_list)):
                        ind = atfp_list.index(triple1)
                        atfp_df[triple1] = atfp_df.at[0, triple1] + 0.5/nat
                    elif (any (fp == triple2 for fp in atfp_list)):
                        ind = atfp_list.index(triple2)
                        atfp_df[atfp_list[ind]] = atfp_df[atfp_list[ind]] + (0.5/nat)
                    elif (not any (fp == triple1 for fp in atfp_list)) and (not any (fp == triple2 for fp in atfp_list)) :
                        ind = len(atfp_list)
                        atfp_list.append(triple1)
                        atfp_df.at[0,atfp_list[ind]] = 0.0
                        atfp_df.at[0,atfp_list[ind]] = atfp_df.at[0,atfp_list[ind]] + (0.5/nat)

    #print (atfp_df)
    return message, atfp_list, atfp_df

