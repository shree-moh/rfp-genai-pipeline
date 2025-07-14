import faiss
import numpy as np
import json
import os

def to_gpu_index(cpu_index, gpu_id=0):
    """
    Move a FAISS index from CPU to GPU for faster search.
    """
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    return gpu_index

def load_vector_db(index_path, texts_path, use_gpu=True, gpu_id=0):
    """
    Loads the FAISS index and document texts.
    Optionally moves the index to GPU.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(texts_path):
        raise FileNotFoundError(f"Document texts file not found: {texts_path}")

    index = faiss.read_index(index_path)
    if use_gpu:
        try:
            index = to_gpu_index(index, gpu_id)
        except Exception as e:
            print(f"Warning: Could not move index to GPU. Using CPU instead. Reason: {e}")

    with open(texts_path, 'r', encoding='utf-8') as f:
        doc_texts = json.load(f)
    return index, doc_texts

def search_vector_db(query_vec, faiss_index, doc_texts, top_k=5):
    """
    Search the FAISS index for the top_k most similar vectors to query_vec.
    Returns the corresponding document texts.
    """
    # Ensure query_vec is a 2D numpy array (shape: [1, embedding_dim])
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    D, I = faiss_index.search(query_vec.astype('float32'), top_k)
    return [doc_texts[i] for i in I[0]]

# Example usage (remove or comment out in production)
if __name__ == "__main__":
    index_path = r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\vector_db\faiss_index.bin'
    texts_path = r'C:\Users\aiselab\Desktop\SAINT\SAINT_INT_jou-master\EHTN_RFP_AGENT\data\doc_texts.json'
    faiss_index, doc_texts = load_vector_db(index_path, texts_path, use_gpu=False)
    # Example: create a dummy query vector with the same dimension as your embeddings
    dummy_query = np.random.rand(1, faiss_index.d).astype('float32')
    results = search_vector_db(dummy_query, faiss_index, doc_texts, top_k=3)
    print("Top results:", results)
