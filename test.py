import pandas as pd
import numpy as np
import joblib
from RDKit_Morgan_String import Morgan_array
from TargetProtein import get_protein_embedding

#Load the trained model
model = joblib.load('xgboost_pIC50.pkl')
drug_pca = joblib.load('drug_pca.pkl')
prot_pca = joblib.load('protein_pca.pkl')
drug_scaler = joblib.load('drug_scaler.pkl')  
prot_scaler = joblib.load('protein_scaler.pkl')

#
test_data = pd.read_csv("test.csv")#temp idk what this is
drug_structs = test_data["SMILES"].tolist()
prot_seqs = test_data["amino_acid_sequence"].tolist()

#Encode & Transform (Must match training steps exactly)
encoded_drug = Morgan_array(drug_structs)
encoded_prot = np.array([get_protein_embedding(s) for s in prot_seqs])

#Transform
scaled_d = drug_pca.transform(encoded_drug)
scaled_p = prot_pca.transform(encoded_prot)
final_d = drug_pca.transform(scaled_d)
final_p = prot_pca.transform(scaled_p)



# Recreate the interaction term (X)
X_test = np.concatenate([final_d, final_p, final_d * final_p], axis=1)

# Predict
predictions = model.predict(X_test)
test_data['predicted_pIC50'] = predictions
test_data.to_csv("results.csv", index=False)