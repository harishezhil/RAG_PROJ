from models.embedding_model import embed_text
import numpy as np
import faiss




def retrieve_answers(query, index, metadata, k=8):  # Increased k
    q_emb = embed_text([query])[0].astype("float32")
    D, I = index.search(np.array([q_emb]), k)
    retrieved = [metadata[i] for i in I[0]]
    return retrieved
