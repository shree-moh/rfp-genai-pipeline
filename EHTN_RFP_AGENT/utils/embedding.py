from sentence_transformers import SentenceTransformer
import torch

def load_embedding_model(model_name):
    """
    Loads a pre-trained SentenceTransformer model onto GPU if available.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer(model_name, device=device)

def embed_query(query, model):
    """
    Generates an embedding for a given query using the specified model.
    """
    return model.encode([query])
