from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ollama import Client
import json


def chunk_text(text, tokenizer, max_tokens=8000, overlap=200):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # Overlapping context
    return chunks


def deduplicate_fuzzy(list, threshold=80):
    unique = []
    for name in list:
        if all(fuzz.ratio(name, existing) < threshold for existing in unique):
            unique.append(name)
    return unique
    
def query_model_return_list(model,chat,tokenizer,local=False):
    formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    device= "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([formatted], return_tensors="pt").to(device)

    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
    }

    outputs = model.generate(**inputs, **gen_kwargs)

    # Strip input prefix and parse
    output_ids = outputs[0][inputs.input_ids.shape[-1]:].tolist()
    # detect </think> token id if present
    try:
        think_idx = len(output_ids) - output_ids[::-1].index(tokenizer.convert_tokens_to_ids('</think>'))
    except ValueError:
        think_idx = 0
    raw = tokenizer.decode(output_ids[think_idx:], skip_special_tokens=True).strip()
    # Parse JSON list
    try:
        answer_list = json.loads(raw)
    except json.JSONDecodeError:
        # fallback to Python literal_eval
        from ast import literal_eval
        try:
            answer_list = literal_eval(raw)
        except Exception:
            answer_list = []

    return answer_list