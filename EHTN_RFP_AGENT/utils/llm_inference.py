from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llama_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

def classify_requirements(context, user_query, tokenizer, model, device, max_new_tokens=512):
    prompt = (
        f"Given the following RFP context:\n{context}\n"
        f"Classify the requirements and assign each to the appropriate department. "
        f"Return as a table with columns: Requirement, Department, Confidence."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
