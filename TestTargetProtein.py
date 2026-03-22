from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

#puts the model into evaluation mode
model.eval()
print('allo ')
def get_protein_embedding(sequence):
    # ProtTrans formatting 
    sequence_spaced = " ".join(list(sequence))
    
    # Tokenize
    ids = tokenizer.batch_encode_plus([sequence_spaced], add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Generate hidden states without calculating gradients 
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Use the 'last_hidden_state' 
    # We take the mean across the sequence length (dim=1) to get a single vector per protein
    features = embedding.last_hidden_state.cpu().numpy()
    protein_vector = np.mean(features, axis=1)
    
    return protein_vector.flatten()

# Example: Sequence for a small peptide
test_seq = "MSLGVASVSIR"
embedding = get_protein_embedding(test_seq)
print(f"Embedding shape: {embedding.shape}") # expect-> 1024-dimensional
