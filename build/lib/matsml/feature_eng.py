""" 
    Huan Tran (huantd@gmail.com)
  
    Feature engineering module

"""

import numpy as np
import pandas as pd


from matsml.data import ProcessData
from matsml.io import AtomicStructure
from matsml.io import goodbye, progress_bar, get_struct_params
import os,math,sys

class FeatureEng:
    """ 
    Fingerprint: compute materials fingerprints from atomic structures

      - data_params: Dictionary, containing parameters needed, see manual

    """

    def __init__(self,data_params):

        # Threadhold of fingerprint components
        self.thld = 1E-8

        print ('  Feature engineering ')
        print ('    summary'.ljust(32),self.summary)


    def read_input(self):
        """
        Functionality: 
            read the input file

        """

        print ('  Read input')
        data_tmp = pd.read_csv(self.summary,delimiter=',')
        self.structs = data_tmp[data_tmp.target.notnull()]

        print ('    num_structs'.ljust(32),len(self.structs))


    def get_fingerprint(self):
        """
        Functionality: 
            Wrapper, compute materials fingerprints

        """
        
        if self.fp_type == 'random_forest':
            fp=self.get_pcm()

        # Remove constant columns
        fp=fp.loc[:,(fp!=fp.iloc[0]).any()]

        # Save
        fp.to_csv(self.fp_file,index=False)
        print ('  Done fingerprinting, results saved in %s'%(self.fp_file))


    def get_soap_dsc(self):
        from dscribe.descriptors import SOAP
        from ase import io
        """
        Functionality: 
            Compute SOAP for molecules and crystals using DScribe

        """
        
        self.read_input()
        struct_df = self.structs
        n_atoms_max, nspecs, species = get_struct_params(self.data_loc, struct_df)

        rcut=8.0
        nmax=7
        lmax=6

        periodic=True if self.fp_type=='soap_crystals' else False

        average_soap=SOAP(species = species, rcut = rcut, nmax = nmax,
            lmax = lmax,periodic = periodic, average="inner", sparse = False)

        fp_dim=average_soap.get_number_of_features()
        columns=['id','target']+['soap_'+str(i).zfill(4) for i in \
            range(fp_dim)]
        soap=pd.DataFrame(columns=columns)

        print ('  Computing SOAP fingerprint with DScribe')

        for k, v in struct_df['file_name'].items():
            sys.stdout.write('\r')
            if self.verbosity==1:
                print ('    processing file %s ' %v)
            elif self.verbosity==0:
                progress_bar(k,len(struct_df),'update')

            idx=np.array(struct_df.loc[struct_df['file_name']==str(v)].\
                index)[0]

            target=np.array(struct_df.loc[struct_df['file_name']==\
                str(v)]['target']).astype(float)[0]

            # SOAP with DScribe 
            struct=io.read(os.path.join(self.data_loc,str(v)))
            this_soap=[str(v),str(target)]+list(average_soap.create(struct)*\
                len(struct))

            soap=soap.append(pd.DataFrame(np.array(this_soap).\
                reshape((1,fp_dim+2)),columns=columns))

        if self.verbosity==0:
            sys.stdout.write('\n')

        return soap

