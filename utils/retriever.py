import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, k=3):
    question_embedding = embedding_model.encode([question], show_progress_bar=False).astype("float32")
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]
