from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("schemes_index.faiss")
metadata = pd.read_csv("scheme_metadata.csv")

def retrieve_schemes(query, top_k=5): 
    
    query_embedding = model.encode([query])
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    results = metadata.iloc[indices[0]]
    return results[["Scheme Name", "Objective", "Key Benefits", "Eligibility", "Source"]]

# Example usage
query = "What schemes are available for small farmers in Maharashtra?"
results = retrieve_schemes(query)
print(results)
