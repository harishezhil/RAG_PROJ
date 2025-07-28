# ✅ Improved utils.py with section chunking and FAISS integration
def section_chunking(text):
    import re
    pattern = r'(?=\n\s*(\*\*|\d+\.))'  # Lookahead for markdown-style headers
    sections = re.split(pattern, text)

    chunks = []
    buffer = ""

    for sec in sections:
        if sec.strip():
            if len(buffer) + len(sec) < 1000:
                buffer += sec
            else:
                chunks.append(buffer.strip())
                buffer = sec
    if buffer:
        chunks.append(buffer.strip())

    return chunks


def load_documents(folder="data"):
    import os, json, pandas as pd
    import xml.etree.ElementTree as ET
    import fitz

    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        content = ""

        if file.endswith(".txt"):
            with open(path, encoding='utf-8') as f:
                content = f.read()

        elif file.endswith(".json"):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

                def flatten_json(obj):
                    if isinstance(obj, dict):
                        return " ".join(flatten_json(v) for v in obj.values())
                    elif isinstance(obj, list):
                        return " ".join(flatten_json(i) for i in obj)
                    else:
                        return str(obj)

                content = flatten_json(data)

        elif file.endswith(".xml"):
            tree = ET.parse(path)
            root = tree.getroot()
            content = " ".join(elem.text for elem in root if elem.text)

        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)
            content = " ".join(df.astype(str).values.flatten())
            
        elif file.endswith(".pdf"):
            doc = fitz.open(path)
            content = ""
            for page in doc:
                content += page.get_text()

        if content:
            docs.append((file, content))

    return docs


def build_faiss_index(documents, index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
    import os, pickle
    from models.embedding_model import embed_text
    import faiss

    os.makedirs("faiss_index", exist_ok=True)
    chunks, metadata = [], []

    for filename, doc_text in documents:
        parts = section_chunking(doc_text)
        chunks.extend(parts)
        metadata.extend([{"chunk": part, "source": filename} for part in parts])

    #Save chunks to a text file for inspection    
    with open("chunks_output.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"[Chunk {i+1} from {metadata[i]['source']}]\n{chunk.strip()}\n\n{'-'*80}\n\n")

    embeddings = embed_text(chunks)
    dim = embeddings[0].shape[0]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index(index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
    import pickle, faiss
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata



# import os
# import json
# import pickle
# import pandas as pd
# import xml.etree.ElementTree as ET
# from models.embedding_model import embed_text
# import faiss


# # def section_chunking(text, chunk_size=500, overlap=100):
# #     chunks = []
# #     for i in range(0, len(text), chunk_size - overlap):
# #         chunks.append(text[i:i + chunk_size])
# #     return chunks


# import re

# def section_chunking(text):
#     # This assumes your text uses markdown-like section headers: ** or numbered bullets
#     pattern = r'(?=\n\s*(\*\*|\d+\.))'  # Lookahead for new sections
#     sections = re.split(pattern, text)
    
#     chunks = []
#     buffer = ""
    
#     for sec in sections:
#         if sec.strip():
#             if len(buffer) + len(sec) < 1000:
#                 buffer += sec
#             else:
#                 chunks.append(buffer.strip())
#                 buffer = sec
#     if buffer:
#         chunks.append(buffer.strip())

#     return chunks


# # Load documents and include filename with content
# def load_documents(folder="data"):
#     docs = []
#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)
#         content = ""

#         if file.endswith(".txt"):
#             with open(path, encoding='utf-8') as f:
#                 content = f.read()

#         # elif file.endswith(".json"):
#         #     with open(path, encoding='utf-8') as f:
#         #         data = json.load(f)
#         #         content = " ".join(str(v) for v in data.values())

#         elif file.endswith(".json"):
#             with open(path, encoding='utf-8') as f:
#                 data = json.load(f)

#                 def flatten_json(obj):
#                     if isinstance(obj, dict):
#                         return " ".join(flatten_json(v) for v in obj.values())
#                     elif isinstance(obj, list):
#                         return " ".join(flatten_json(i) for i in obj)
#                     else:
#                         return str(obj)

#                 content = flatten_json(data)

#         elif file.endswith(".xml"):
#             tree = ET.parse(path)
#             root = tree.getroot()
#             content = " ".join(elem.text for elem in root if elem.text)

#         elif file.endswith(".xlsx"):
#             df = pd.read_excel(path)
#             content = " ".join(df.astype(str).values.flatten())

#         if content:
#             docs.append((file, content))  # (filename, document_text)

#     return docs


# # Build FAISS index with metadata tracking (chunk_text + source_filename)
# def build_faiss_index(documents, index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
#     os.makedirs("faiss_index", exist_ok=True)
#     chunks, metadata = [], []

#     for filename, doc_text in documents:
#         parts = section_chunking(doc_text)
#         chunks.extend(parts)
#         metadata.extend([{"chunk": part, "source": filename} for part in parts])

#     embeddings = embed_text(chunks)
#     dim = embeddings[0].shape[0]
#     print(f"Embedding dimension: {dim}")

#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     faiss.write_index(index, index_path)
#     with open(meta_path, "wb") as f:
#         pickle.dump(metadata, f)


# # Load FAISS index and metadata
# def load_index(index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
#     index = faiss.read_index(index_path)
#     with open(meta_path, "rb") as f:
#         metadata = pickle.load(f)
#     return index, metadata
