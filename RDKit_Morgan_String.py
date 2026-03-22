import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
#add numpy

# Read a CSV file


#def Morgan_array():



def Morgan_array(smiles_list: list):

    fingerprints = []

    for smile in smiles_list:
        fp = Morgan_Bitstring(smile)        # Generate the Morgan fingerprint for the current SMILES string using RDKit
        print (smile)
        fingerprints.append(fp)             # Append the generated fingerprint to the list of fingerprints

    return np.array(fingerprints)           # Return the numerical list of fingerprints as a NumPy array


def Morgan_Bitstring (SMILE_STRING: str):       # temps function to get the bitstrings from the SMILE code
# Create a bitstring from a SMILES string

    arr = np.zeros((2048,), dtype=int)

    mol = Chem.MolFromSmiles(SMILE_STRING)
    #maybe handle error (if its invalid for example)
    #Generate Morgan fingerprint (ECFP-like)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    DataStructs.ConvertToNumpyArray(fp, arr)             #convert bitstring into numerical array

    return arr  

datafile = pd.read_csv("train.csv",nrows=100)
drug_structs=datafile["SMILES"].tolist()
encoded_drug=Morgan_array(drug_structs)                  #this can be tuned according to the size of the dataset, for testing purposes we will use 100


