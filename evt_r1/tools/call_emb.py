from sentence_transformers import SentenceTransformer
import torch

text_emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

def call_text_emb_model(completions):
    query_embeddings = text_emb_model.encode(completions)
    response_tensors = torch.tensor(query_embeddings, dtype=torch.float32)
    return response_tensors