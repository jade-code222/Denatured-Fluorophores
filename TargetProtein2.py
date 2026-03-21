import pandas as pd
import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import pickle

# 1. Setup (The part you already have)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
model.eval()

def get_protein_embedding(sequence):
    sequence_spaced = " ".join(list(sequence))
    ids = tokenizer.batch_encode_plus([sequence_spaced], add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    features = embedding.last_hidden_state.cpu().numpy()
    return np.mean(features, axis=1).flatten()

# --- 2. THE PART THAT READS YOUR FILE ---
print("Reading train.csv...")
df = pd.read_csv('train.csv')

# Optimization: Only embed each unique sequence ONCE [cite: 63]
unique_seqs = df['amino_acid_sequence'].unique() 
print(f"Found {len(unique_seqs)} unique protein sequences.")

protein_lookup = {}

for i, seq in enumerate(unique_seqs):
    # This is where the actual 'reading' and 'calculating' happens
    protein_lookup[seq] = get_protein_embedding(seq)
    
    if i % 10 == 0:
        print(f"Processed {i}/{len(unique_seqs)} proteins...")

# 3. Save it so you never have to do this again
with open('protein_embeddings.pkl', 'wb') as f:
    pickle.dump(protein_lookup, f)

print("Finished! Your protein data is now saved in protein_embeddings.pkl")