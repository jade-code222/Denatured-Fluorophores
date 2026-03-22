#the model will be trained on X = [Morgan_fp + Prot_embedding] y = pIC50
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from RDKit_Morgan_String import Morgan_array
from TargetProtein import get_protein_embedding
from model import xgboost_model, best_model

#Initialize dataset
datafile = pd.read_csv("train.csv")
drug_structs=datafile["SMILES"].tolist()
protein_structs=datafile["amino acid sequence"].tolist()
y=datafile["pIC50"].values

#call the encoders (morgan fp, protein)
encoded_drug=Morgan_array(datafile)
encoded_features=get_protein_embedding(protein_structs)
#call interaction builder

#train model
R2,MSE,MAE=xgboost_model(X,y)
print(f"Metrics:R2={R2:.3f}, MSE={MSE:.3f}, MAE={MAE:.3f}")

#Fine tuning
best_R2_depth, best_MSE_depth, best_MAE_depth = best_model(X, y)