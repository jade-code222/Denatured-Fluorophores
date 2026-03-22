import pandas as pd
import numpy as np
import joblib
from RDKit_Morgan_String import Morgan_array
from TargetProtein import get_protein_embedding

#Load the trained model
model = joblib.load('xgboost_pIC50.pkl')
drug_pca = joblib.load('drug_pca.pkl')
prot_pca = joblib.load('protein_pca.pkl')

#
test_data = pd.read_csv("test.csv")
drug_structs = test_data["SMILES"].tolist()
prot_seqs = test_data["amino acid sequence"].tolist()

#Encode & Transform (Must match training steps exactly)
encoded_drug = Morgan_array(drug_structs)
encoded_prot = np.array([get_protein_embedding(s) for s in prot_seqs])

# USE TRANSFORM ONLY (Don't re-fit the PCA)
drug_red = drug_pca.transform(encoded_drug)
prot_red = prot_pca.transform(encoded_prot)

# Recreate the interaction term (X)
X_test = np.concatenate([drug_red, prot_red, drug_red * prot_red], axis=1)

# Predict
predictions = model.predict(X_test)
test_data['predicted_pIC50'] = predictions
test_data.to_csv("results.csv", index=False)