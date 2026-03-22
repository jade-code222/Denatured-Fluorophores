#the model will be trained on X = [Morgan_fp + Prot_embedding] y = pIC50
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from RDKit_Morgan_String import Morgan_array
from TargetProtein import get_protein_embedding
from model import xgboost_model, best_model
import joblib

#Initialize dataset
datafile = pd.read_csv("train.csv")
drug_structs=datafile["SMILES"].tolist()
protein_structs=datafile["amino acid sequence"].tolist()
y=datafile["pIC50"].values

#call the encoders (morgan fp, protein)
encoded_drug=Morgan_array(drug_structs)
encoded_protein = np.array([get_protein_embedding(seq) for seq in protein_structs]) 

#dimentionality reduction by PCA
drug_PCA=PCA(n_components=200)#this can be tuned
drug_reduced = drug_PCA.fit_transform(encoded_drug)

pca_protein = PCA(n_components=200)#this can be tuned
protein_reduced = pca_protein.fit_transform(encoded_protein)


#concatenate and add the element cross product to provide prossiblity of interation
X= np.concatenate([drug_reduced, protein_reduced, drug_reduced*protein_reduced],axis=1)

#train model
R2,MSE,MAE=xgboost_model(X,y)
print(f"Metrics:R2={R2:.3f}, MSE={MSE:.3f}, MAE={MAE:.3f}")

#Fine tuning
best_R2_depth, best_MSE_depth, best_MAE_depth = best_model(X, y)

best_depth = best_R2_depth 

# FINAL STEP: Train on 100% of data with best depth
final_model = XGBRegressor(n_estimators=300, max_depth=best_depth, learning_rate=0.1)
final_model.fit(X, y)

# Save everything
joblib.dump(final_model, 'xgboost_pIC50.pkl')
joblib.dump(drug_PCA, 'drug_pca.pkl')
joblib.dump(pca_protein, 'protein_pca.pkl')