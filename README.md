RFP AI Agent
Automated RFP Requirement Classification & Reporting with Vector Search and LLMs

Overview
RFP AI Agent is a modular pipeline that leverages vector search and large language models (LLMs) to automatically classify requirements in Request for Proposal (RFP) documents and generate structured Excel reports.
It combines document embeddings, FAISS vector search, and state-of-the-art LLMs (like Llama 2 or Mistral) for fast, accurate, and scalable RFP analysis.

Features
🔍 Semantic Search: Retrieve relevant requirements using the FAISS vector database.

🤖 LLM-Powered Classification: Classify and extract structured information from RFPs with Llama 2, Mistral, or similar models.

📊 Automated Reporting: Generate Excel reports for easy review and downstream processing.

🛠️ Modular Design: Easy to extend, adapt, and integrate into enterprise workflows.

1. Clone the Repository
bash
git clone https://github.com/yourusername/rfp-ai-agent.git
cd rfp-ai-agent
2. Set Up the Environment
bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
3. Prepare Your Data
Edit  data/doc_texts.json with your RFP or requirement text chunks (as a JSON array).

4. Generate Embeddings
bash
python utils/build_embeddings.py
5. Build the FAISS Index
bash
python utils/build_faiss_index.py
6. Run the Main Pipeline
bash
python main.py
Enter your RFP query when prompted.

The pipeline retrieves relevant requirements, classifies them, and generates RFP_Classification_Report.xlsx.

Configuration
Edit config.py to set:

Embedding model name (e.g., 'all-MiniLM-L6-v2')

LLM model name (e.g., 'meta-llama/Llama-2-7b-chat-hf' or 'mistralai/Mistral-7B-Instruct-v0.2')

Paths for FAISS index and document texts

Requirements
Python 3.8+

sentence-transformers

faiss-cpu (use CPU version on Windows)

transformers

openpyxl

huggingface_hub

See requirements.txt for the full list.

Notes
Model Access:
Some LLMs (Llama 2, Mistral) require requesting access and logging in via huggingface-cli login.

FAISS GPU:
On Windows, use faiss-cpu. GPU support is available on Linux.

Customization:
You can swap in any embedding model or LLM supported by Hugging Face.

License
This project is licensed under the Apache 2.0 License.

Acknowledgements
Meta AI for Llama 2

Mistral AI for Mistral-7B

Hugging Face for model hosting and ecosystem
