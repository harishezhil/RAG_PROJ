from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts):
 
    #Embed a list of texts and return their vector representations.
    
    return model.encode(texts, convert_to_tensor=False)
