from sentence_transformers import SentenceTransformer
import numpy as np
import faiss   
import sqlite3
import pandas as pd

conn = sqlite3.connect("schemes.db")
df = pd.read_sql_query("SELECT * FROM schemes_data", conn)
conn.close()

print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns.tolist())
print("First few rows:\n", df.head())

if df.empty:
    raise ValueError("No data found in 'schemes.db'. Ensure the 'schemes_data' table is populated.")

required_columns = ["Objective", "Key Benefits", "Eligibility"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame. Check your database schema.")
df["combined_text"] = df[required_columns].fillna("No information available").agg(" ".join, axis=1)

print("Combined text sample:\n", df["combined_text"].head())
if df["combined_text"].str.strip().eq("").all():
    raise ValueError("All combined_text entries are empty. Check your data content.")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)
if embeddings.shape[0] == 0:
    raise ValueError("No embeddings generated. Input data might be invalid.")

# Save embeddings to FAISS index
dimension = embeddings.shape[1]  # Get the embedding size (e.g., 384 for MiniLM)
print(f"Embedding dimension: {dimension}")
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
index.add(embeddings)  
faiss.write_index(index, "schemes_index.faiss")

# Save scheme metadata for retrieval
metadata_columns = ["Scheme Name", "Objective", "Key Benefits", "Eligibility", "Source"]
df[metadata_columns].to_csv("scheme_metadata.csv", index=False)

print("Embeddings generated and stored in 'schemes_index.faiss'")
