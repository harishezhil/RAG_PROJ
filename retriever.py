from models.embedding_model import embed_text
import numpy as np
import faiss

import re

def extract_years(text):
    return re.findall(r"\b(20\d{2})\b", text)

def retrieve_answers(query, index, metadata, k=5):
    from langchain.schema import Document
    
    # Extract years from query
    query_years = extract_years(query)

    # Perform initial similarity search
    results = index.similarity_search(query, k=15)  # pull more, we'll filter below

    if query_years:
        filtered = []
        for doc in results:
            doc_years = extract_years(doc.page_content)
            if any(year in doc_years for year in query_years):
                filtered.append(doc)
        if filtered:
            return filtered[:k]  # return top-k matching
        else:
            return results[:k]  # fallback if no year match
    else:
        return results[:k]



# def retrieve_answers(query, index, metadata, k=8):  # Increased k
#     q_emb = embed_text([query])[0].astype("float32")
#     D, I = index.search(np.array([q_emb]), k)
#     retrieved = [metadata[i] for i in I[0]]
#     return retrieved
