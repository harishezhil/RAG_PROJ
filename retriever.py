from models.embedding_model import embed_text
import numpy as np
import faiss
import re

def extract_years(text):
    return re.findall(r"\b(20\d{2})\b", text)

def retrieve_answers(query, index, metadata, k=5):
    from langchain.schema import Document

    query_years = extract_years(query)
    query_lower = query.lower()

    # Over-fetch to allow better filtering
    results = index.similarity_search(query, k=20)

    # Optional: Force insert critical fact-based chunks (e.g., founders)
    founding_keywords = ["found", "founder", "create", "establish", "start"]
    should_force_founding_chunk = any(keyword in query_lower for keyword in founding_keywords)

    forced_chunk = None
    if should_force_founding_chunk:
        for meta in metadata:
            if "Flipkart was founded" in meta.page_content:
                forced_chunk = meta
                break

    # Filter by year match if applicable
    if query_years:
        filtered = []
        for doc in results:
            doc_years = extract_years(doc.page_content)
            if any(year in doc_years for year in query_years):
                filtered.append(doc)

        final = filtered[:k]
    else:
        final = results[:k]

    # Inject forced chunk if it's not already included
    if forced_chunk and all(forced_chunk.page_content != d.page_content for d in final):
        final = [forced_chunk] + final[:k-1]  # Ensure only k total

    return final



# from models.embedding_model import embed_text
# import numpy as np
# import faiss

# import re

# def extract_years(text):
#     return re.findall(r"\b(20\d{2})\b", text)

# def retrieve_answers(query, index, metadata, k=5):
#     from langchain.schema import Document
    
#     # Extract years from query
#     query_years = extract_years(query)

#     # Perform initial similarity search
#     results = index.similarity_search(query, k=15)  # pull more, we'll filter below

#     if query_years:
#         filtered = []
#         for doc in results:
#             doc_years = extract_years(doc.page_content)
#             if any(year in doc_years for year in query_years):
#                 filtered.append(doc)
#         if filtered:
#             return filtered[:k]  # return top-k matching
#         else:
#             return results[:k]  # fallback if no year match
#     else:
#         return results[:k]



# # def retrieve_answers(query, index, metadata, k=8):  # Increased k
# #     q_emb = embed_text([query])[0].astype("float32")
# #     D, I = index.search(np.array([q_emb]), k)
# #     retrieved = [metadata[i] for i in I[0]]
# #     return retrieved
