""" 
    Huan Tran (huantd@gmail.com)
  
    Fingerprint module: fingerprint/featurize molecules and crystals in 
        some ways
      - Molecules: projected Coulomb matrix, SOAP
      - Crystal: projected Ewald sum matrix, SOAP

"""

import numpy as np
import pandas as pd
from matsml.data import ProcessData
from matsml.io import AtomicStructure
from matsml.io import goodbye, progress_bar, get_struct_params
import os
import sys
import math


class Fingerprint:
    """ 
    Fingerprint: compute materials fingerprints from atomic structures
      - data_params: Dictionary, containing parameters needed, see manual

    """

    def __init__(self, data_params):

        # Threadhold of fingerprint components
        self.thld = 1E-8

        self.data_params = data_params
        self.fp_type = self.data_params['fp_type']
        self.summary = self.data_params['summary']
        self.data_loc = self.data_params['data_loc']
        self.fp_file = self.data_params['fp_file']
        self.fp_dim = self.data_params['fp_dim']
        self.verbosity = self.data_params['verbosity']

        print('  Atomic structure fingerprinting')
        print('    summary'.ljust(32), self.summary)
        print('    data_loc'.ljust(32), self.data_loc)
        print('    fp_type'.ljust(32), self.fp_type)
        print('    fp_file'.ljust(32), self.fp_file)
        print('    fp_dim'.ljust(32), self.fp_dim)
        print('    verbosity'.ljust(32), self.verbosity)

    def read_input(self):
        """
        Functionality: 
            read the input file

        """

        print('  Read input')
        data_tmp = pd.read_csv(self.summary, delimiter=',')
        self.structs = data_tmp[data_tmp.target.notnull()]

        print('    num_structs'.ljust(32), len(self.structs))

    def gaussian(self, x, sigma, r):
        """ 
        Functionality: 
            return Gaussian function 

        """
        return 1./(math.sqrt(sigma**math.pi))*np.exp(-sigma*np.power((x-r), 2.))

    def get_fingerprint(self):
        """
        Functionality: 
            Wrapper, compute materials fingerprints

        """

        if self.fp_type == 'pcm_molecs_matsml':
            fp = self.get_pcm()
        elif self.fp_type == 'pcm_molecs':
            fp = self.get_pcm_dsc()
        elif self.fp_type == 'soap_molecs':
            fp = self.get_soap_dsc()
        elif self.fp_type == 'soap_crystals':
            fp = self.get_soap_dsc()
        elif self.fp_type == 'pesm_crystals':
            fp = self.get_pesm_dsc()

        # Remove constant columns
        fp = fp.loc[:, (fp != fp.iloc[0]).any()]

        # Save
        fp.to_csv(self.fp_file, index=False)
        print('  Done fingerprinting, results saved in %s' % (self.fp_file))

    def get_pcm(self):
        """
        Functionality: 
            Compute the projected coulomb matrix of molecules

        Note: this method is much faster than "get_pcm_dsc", which uses dscribe
              for creating CM and ASE to import structure. 

        """

        self.read_input()
        ats = AtomicStructure()
        fp_dim = self.fp_dim
        struct_df = self.structs

        AtomSpecs = {'specs': ['C', 'H', 'O', 'F', 'N', 'S', 'Cl'],
                     'znucl': [6, 1, 8, 9, 7, 16, 17]}
        specs_df = pd.DataFrame(AtomSpecs, columns=['specs', 'znucl'])

        columns = ['id', 'target'] + \
            ['pcm_'+str(i).zfill(4) for i in range(fp_dim)]
        pcm = pd.DataFrame(columns=columns)

        print('  Computing Coulomb matrix')

        cm_all = []
        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity == 1:
                print('    processing file %s ' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            nat, nspecs, specs, xyz_df = ats.read_xyz(os.path.join(self.data_loc,
                                                                   str(v)))

            cm = np.zeros((nat, nat))

            for iat in range(nat-1):
                xcart = xyz_df.iloc[iat][['x', 'y', 'z']]
                xi = np.array(xcart).astype(float)
                spi = np.array(xyz_df.iloc[iat][['specs']]).astype(str)[0]
                zi = np.array(specs_df.loc[specs_df['specs'] == str(spi)]
                              ['znucl']).astype(float)[0]
                for jat in range(iat, nat, 1):
                    spj = np.array(xyz_df.iloc[jat][['specs']]).astype(str)[0]
                    zj = np.array(specs_df.loc[specs_df['specs'] == str(spj)]
                                  ['znucl']).astype(float)[0]
                    xcart = xyz_df.iloc[jat][['x', 'y', 'z']]
                    xj = np.array(xcart).astype(float)
                    if iat == jat:
                        cm[iat][jat] = 0.5*zi**2.4
                    else:
                        cm[iat][jat] = zi*zj/np.linalg.norm(xi-xj)

            cm_all.append(cm.flatten().astype(list))

        if self.verbosity == 0:
            sys.stdout.write('\n')

        # Projected Coulomb matrix
        print('  Projecting Coulomb matrix to create fingerprints')
        xmin = min([min(r) for r in cm_all])
        xmax = min([max(r) for r in cm_all])
        ri = np.linspace(xmin, xmax, fp_dim)
        sigma = 1.5*(xmax-xmin)/fp_dim

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            idx = np.array(struct_df.loc[struct_df['file_name'] == str(v)].
                           index)[0]

            if self.verbosity == 1:
                print('    processing entries %s of the CM' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            target = np.array(
                struct_df.loc[
                    struct_df['file_name'] == str(v)
                ]['target']
            ).astype(float)[0]

            cm_flatten = cm_all[idx]
            gcm = [str(v), str(target)]

            for i in range(fp_dim):
                gcm_el = sum(
                    self.gaussian(
                        cm_flatten[j], 
                        sigma, 
                        ri[i]
                    ) for j in range(len(cm_flatten))
                )

                if np.absolute(gcm_el) < self.thld:
                    gcm_el = 0.0
                gcm.append(gcm_el)

            pcm = pd.concat(
                [
                    pcm, 
                    pd.DataFrame(
                        np.array(gcm).reshape((1, fp_dim+2)), 
                        columns=columns
                    )
                ],
                axis = 0
            )

        if self.verbosity == 0:
            sys.stdout.write('\n')

        return pcm

    def get_pesm_dsc(self):
        from dscribe.descriptors import EwaldSumMatrix
        from ase import io

        """
        Functionality: 
            Compute the projected Ewald sum matrix of crystals using DScribe

        Note:
            Original reference
                F. Faber, A. Lindmaa, O. Anatole von Lilienfeld, & R. Armiento, 
                    "Crystal structure representations for machine learning 
                    models of formation energies", Int. J. Quantum Chem. 115, 
                    1094–1101 (2015).
            Implementation
                L. Himanen, M.O.J. Jäger, E. V.Morooka, F. F. Canova, Y. S.
                    Ranawat, D. Z. Gao, P. Rinke, and A. S.Foste, "DScribe: 
                    Library of descriptors for machine learning in materials 
                    science, Comput. Phys. Commun. 247, 106949 (2020)
                https://singroup.github.io/dscribe/latest/

        """

        self.read_input()
        fp_dim = self.fp_dim
        struct_df = self.structs
        n_atoms_max, nspecs, species = get_struct_params(
            self.data_loc, struct_df)

        # Ewald summation parameters
        r_cut = 4
        g_cut = 4

        columns = ['id', 'target'] + \
            ['pesm_'+str(i).zfill(4) for i in range(fp_dim)]
        pesm = pd.DataFrame(columns=columns)

        print('  Computing Ewald sum Matrix')
        esm_all = []

        # Ewald sum matrix with DScribe
        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity == 1:
                print('    processing file %s ' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            crystal = io.read(os.path.join(self.data_loc, str(v)))

            esm = EwaldSumMatrix(
                n_atoms_max=n_atoms_max, 
                permutation="none",
            )

            esm_this = esm.create(
                crystal, 
                r_cut=r_cut, 
                g_cut=g_cut
            )

            esm_all.append(esm_this)

        if self.verbosity == 0:
            sys.stdout.write('\n')

        # Projected Ewald sum matrix
        print('  Projecting Ewald sum matrix to create fingerprints')
        xmin = min([min(r) for r in esm_all])
        xmax = min([max(r) for r in esm_all])
        ri = np.linspace(xmin, xmax, fp_dim)
        sigma = 1.75*(xmax-xmin)/fp_dim

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            # Dont want to use k as index here
            idx = np.array(struct_df.loc[struct_df['file_name'] == str(v)].
                           index)[0]

            if self.verbosity == 1:
                print('    processing entry %s of the Ewald sum matrix' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            target = np.array(struct_df.loc[struct_df['file_name'] ==
                                            str(v)]['target']).astype(float)[0]

            ems_crystal = esm_all[k]

            gcm = [str(v), str(target)]
            for i in range(fp_dim):
                gcm_el = sum(self.gaussian(ems_crystal[j], sigma, ri[i]) for j in
                             range(len(ems_crystal)))
                if np.absolute(gcm_el) < self.thld:
                    gcm_el = 0.0
                gcm.append(gcm_el)

            pesm = pd.concat(
                [
                    pesm, 
                    pd.DataFrame(
                        np.array(gcm).reshape((1, fp_dim+2)),
                        columns=columns
                    )
                ],
                axis = 0
            )
            #pesm = pesm.append(pd.DataFrame(np.array(gcm).reshape((1, fp_dim+2)),
            #                                columns=columns))

        if self.verbosity == 0:
            sys.stdout.write('\n')

        return pesm

    def get_pcm_dsc(self):
        from dscribe.descriptors import CoulombMatrix
        from ase import io
        """
        Functionality: 
            Compute the projected coulomb matrix of molecules using DScribe

        """

        self.read_input()
        fp_dim = self.fp_dim
        struct_df = self.structs
        n_atoms_max, nspecs, species = get_struct_params(
            self.data_loc, struct_df)

        columns = ['id', 'target'] + \
            ['pcm_'+str(i).zfill(4) for i in range(fp_dim)]
        pcm = pd.DataFrame(columns=columns)

        print('  Computing Coulomb matrix')
        cm_all = []

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity == 1:
                print('    processing file %s ' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            # CoulombMatrix with DScribe
            cm = CoulombMatrix(n_atoms_max=n_atoms_max)
            molec = io.read(os.path.join(self.data_loc, str(v)))
            cm_molec = cm.create(molec)
            cm_all.append(cm_molec.flatten().astype(list))

        if self.verbosity == 0:
            sys.stdout.write('\n')

        # Projected Coulomb matrix
        print('  Projecting Coulomb matrix to create fingerprints')
        xmin = min([min(r) for r in cm_all])
        xmax = min([max(r) for r in cm_all])
        ri = np.linspace(xmin, xmax, fp_dim)
        sigma = 1.5*(xmax-xmin)/fp_dim

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity == 1:
                print('    processing entries %s of the CM' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            idx = np.array(struct_df.loc[struct_df['file_name'] == str(v)].
                           index)[0]

            target = np.array(struct_df.loc[struct_df['file_name'] ==
                                            str(v)]['target']).astype(float)[0]

            cm_molec = cm_all[k]

            gcm = [str(v), str(target)]

            for i in range(fp_dim):
                gcm_el = sum(self.gaussian(cm_molec[j], sigma, ri[i]) for j in
                             range(len(cm_molec)))
                if np.absolute(gcm_el) < self.thld:
                    gcm_el = 0.0
                gcm.append(gcm_el)

            pcm = pd.concat(
                [
                    pcm, 
                    pd.DataFrame(
                        np.array(gcm).reshape((1, fp_dim+2)),
                        columns=columns
                    )
                ], 
                axis = 0
            )
            #pcm = pcm.append(pd.DataFrame(np.array(gcm).reshape((1, fp_dim+2)),
            #                              columns=columns))

        if self.verbosity == 0:
            sys.stdout.write('\n')

        return pcm

    def get_soap_dsc(self):
        from dscribe.descriptors import SOAP
        from ase import io
        """
        Functionality: 
            Compute SOAP for molecules and crystals using DScribe

        """

        self.read_input()
        struct_df = self.structs
        n_atoms_max, nspecs, species = get_struct_params(
            self.data_loc, struct_df)

        r_cut = 8.0
        n_max = 7
        l_max = 6

        periodic = True if self.fp_type == 'soap_crystals' else False

        average_soap = SOAP(
                species=species, 
                r_cut=r_cut, 
                n_max=n_max,
                l_max=l_max, 
                periodic=periodic, 
                average="inner", 
                sparse=False
        )

        fp_dim = average_soap.get_number_of_features()
        columns = ['id', 'target']+['soap_'+str(i).zfill(4) for i in
                                    range(fp_dim)]
        soap = pd.DataFrame(columns=columns)

        print('  Computing SOAP fingerprint with DScribe')

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity == 1:
                print('    processing file %s ' % v)
            elif self.verbosity == 0:
                progress_bar(k, len(struct_df), 'update')

            idx = np.array(struct_df.loc[struct_df['file_name'] == str(v)].
                           index)[0]

            target = np.array(struct_df.loc[struct_df['file_name'] ==
                                            str(v)]['target']).astype(float)[0]

            # SOAP with DScribe
            struct = io.read(os.path.join(self.data_loc, str(v)))
            this_soap = [str(v), str(target)]+list(average_soap.create(struct) *
                                                   len(struct))

            soap = pd.concat(
                [soap, pd.DataFrame(
                    np.array(this_soap).reshape(1, fp_dim + 2),
                    columns=columns
                )],
                axis=0
            )

        if self.verbosity == 0:
            sys.stdout.write('\n')

        return soap
