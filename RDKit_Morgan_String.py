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
        fp = Morgan_Bitstring(smile)
        print (smile)
        fingerprints.append(fp)

    return fingerprints

    """
    # Read a CSV file-->will do in main

    df = pd.read_csv("train.csv")

    bitstring_Array = pd.DataFrame({"SMILES": [], "PROTEINS": []})     # creates an empty list to store the bitstrings

    bitstring_Array["SMILES"] = bitstring_Array["SMILES"].astype(object)          # ensures that the "SMILES" column is of type string

    bitstring_Array["PROTEINS"] = bitstring_Array["PROTEINS"].astype(object)      # ensures that the "PROTEINS" column is of type string
    
    for index in range(len(df)//13):                                     # iterates through each row in the DataFrame

        SMILE_STRING = df.loc[index, "SMILES"]                       # retrieves the SMILES string from the current row

        bitstring = Morgan_Bitstring(SMILE_STRING)                  # generates the NUMERICAL bitstring for the current SMILES string

        bitstring_Array.loc[index,"SMILES"] = bitstring              # appends the SMILES bitstring to the bitsting list

        TARGET_PROTEIN = df.loc[index, "amino_acid_sequence"]                   # retrieves the target protein from the current row

        bitstring_Array.loc[index,"PROTEINS"] = TARGET_PROTEIN       # appends the target protein to the bitsting list

    print (bitstring_Array)  # prints the "SMILES" column of the bitstring array

    return bitstring_Array  #fix according to the fact that isnt string
    """


#this should be first?
def Morgan_Bitstring (SMILE_STRING: str):       # temps function to get the bitstrings from the SMILE code
# Create a bitstring from a SMILES string

    arr = np.zeros((2048,), dtype=int)

    mol = Chem.MolFromSmiles(SMILE_STRING)
    #maybe handle error (if its invalid for example)
    #Generate Morgan fingerprint (ECFP-like)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    DataStructs.ConvertToNumpyArray(fp, arr)             #convert bitstring into numerical array

    return arr  

datafile = pd.read_csv("train.csv")
drug_structs=datafile["SMILES"].tolist()
Morgan_array(drug_structs.head(10))
#must be an array of numbers [1,0,1,1,0,...,1]

