import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Combine relevant text fields
df["text"] = df["name"].astype(str) + " " + df["description"].astype(str)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Convert embeddings to numpy array
embeddings = np.array(embeddings, dtype="float32")

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.bin")

# Save dataframe with IDs (for retrieval)
df.to_csv("indexed_data.csv", index=False)

print("Embeddings generated and stored successfully!")
