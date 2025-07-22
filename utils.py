import os
import json
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
from models.embedding_model import embed_text
import faiss


def section_chunking(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# Load documents and include filename with content
def load_documents(folder="data"):
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
                content = " ".join(str(v) for v in data.values())

        elif file.endswith(".xml"):
            tree = ET.parse(path)
            root = tree.getroot()
            content = " ".join(elem.text for elem in root if elem.text)

        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)
            content = " ".join(df.astype(str).values.flatten())

        if content:
            docs.append((file, content))  # (filename, document_text)

    return docs


# Build FAISS index with metadata tracking (chunk_text + source_filename)
def build_faiss_index(documents, index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
    os.makedirs("faiss_index", exist_ok=True)
    chunks, metadata = [], []

    for filename, doc_text in documents:
        parts = section_chunking(doc_text)
        chunks.extend(parts)
        metadata.extend([{"chunk": part, "source": filename} for part in parts])

    embeddings = embed_text(chunks)
    dim = embeddings[0].shape[0]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


# Load FAISS index and metadata
def load_index(index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
