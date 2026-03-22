import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. Load Data and Precalculated Protein Embeddings
print("Loading data...")
df = pd.read_csv('train.csv') # cite: 44
with open('protein_embeddings.pkl', 'rb') as f:
    protein_lookup = pickle.load(f)

# 2. Function to generate Drug Fingerprints (Morgan/Circular)
def get_drug_fp(smiles):
    mol = Chem.MolFromSmiles(smiles) # cite: 149
    if mol:
        # 2048-bit Morgan Fingerprint (Radius 2 is standard)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        # Fallback for invalid SMILES
        return np.zeros(2048)

# 3. Match Protein Embeddings to every row in the CSV
print("Mapping protein embeddings...")
# This aligns the unique embeddings with the 331k rows in your file
protein_features = np.stack(df['amino_acid_sequence'].map(protein_lookup).values)

# 4. Generate Drug Fingerprints for every row
print("Generating drug fingerprints (this may take a few minutes)...")
drug_features = np.stack(df['SMILES'].apply(get_drug_fp).values)

# 5. CONCATENATE: Combine Drug (2048) + Protein (1024)
# This creates the "Protein-Agnostic" interaction vector [cite: 70]
X = np.hstack([drug_features, protein_features])

# 6. Target Variable (pIC50)
y = df['pIC50'].values # cite: 54, 68

print(f"\nFinal Matrix X Shape: {X.shape}") # Expected: (331000, 3072)
print(f"Final Target y Shape: {y.shape}")