import numpy as np
import faiss
import json

# Load embeddings
doc_embeddings = np.load(r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\vector_db\doc_embeddings.npy')

# Load document texts
with open(r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\doc_texts.json', 'r', encoding='utf-8') as f:
    doc_texts = json.load(f)

# Verify embeddings and texts count match
assert len(doc_embeddings) == len(doc_texts), "Embeddings and document texts count mismatch!"

# Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Save FAISS index
faiss.write_index(index, r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\vector_db\faiss_index.bin')

print("FAISS index built and saved successfully.")
