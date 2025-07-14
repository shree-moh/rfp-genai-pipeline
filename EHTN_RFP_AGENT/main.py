from utils.embedding import load_embedding_model, embed_query
from utils.vector_search import load_vector_db, search_vector_db
from utils.llm_inference import load_llama_model, classify_requirements
from utils.parser import parse_llm_output
from utils.reporting import write_to_excel
import config

def main():
    # Load models and data
    embed_model = load_embedding_model(config.EMBEDDING_MODEL_NAME)
    faiss_index, doc_texts = load_vector_db(config.VEC_DB_PATH, config.DOC_TEXTS_PATH)
    tokenizer, llama_model, device = load_llama_model(config.LLAMA_MODEL_NAME)

    # 1. Get user query
    user_query = input("Enter your RFP query: ")

    # 2. Embed and search
    query_vec = embed_query(user_query, embed_model)
    context_chunks = search_vector_db(query_vec, faiss_index, doc_texts)
    context = " ".join(context_chunks)

    # 3. LLM inference
    llm_output = classify_requirements(context, user_query, tokenizer, llama_model, device)

    # 4. Parse output
    structured_data = parse_llm_output(llm_output)

    # 5. Write Excel report
    write_to_excel(structured_data, filename="RFP_Classification_Report.xlsx")

if __name__ == "__main__":
    main()
