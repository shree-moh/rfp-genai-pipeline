from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load document texts
with open(r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\doc_texts.json', 'r', encoding='utf-8') as f:
    doc_texts = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
doc_embeddings = model.encode(doc_texts, show_progress_bar=True)

# Save embeddings
np.save(r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\vector_db\doc_embeddings.npy', doc_embeddings)

print("Embeddings generated and saved to data/vector_db/doc_embeddings.npy")
