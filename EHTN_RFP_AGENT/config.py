
# Embedding model (Hugging Face model name)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# FAISS index and document texts paths
VEC_DB_PATH = 'data/vector_db/faiss_index.bin'
DOC_TEXTS_PATH = 'data/doc_texts.json'

# Llama 2 model (Hugging Face model name)
LLAMA_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'

# (Optional) Add more configuration variables as needed, e.g.:
# Number of top documents to retrieve
TOP_K = 5

# Output Excel report filename (optional, can be set in main.py)
EXCEL_REPORT_FILENAME = 'RFP_Classification_Report.xlsx'
