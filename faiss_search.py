import faiss
import numpy as np
import json
import pandas as pd

# Load product data (assuming you've pre-generated sentence embeddings)
def load_product_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to load sentence embeddings (generated earlier and saved in CSV)
def load_embeddings(file_path):
    return pd.read_csv(file_path)

# Create FAISS index (L2 distance index)
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)  # Use L2 distance metric
    index.add(embeddings)  # Add embeddings to the index
    return index

# Save FAISS index to file
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# Main function to create and save index
def create_and_save_index():
    # Load embeddings (assumes embeddings have already been generated)
    embeddings = load_embeddings('product_embeddings.csv').values[:, 1:].astype(np.float32)  # Skip the ID column
    # Create FAISS index
    index = create_faiss_index(embeddings)
    # Save index
    save_faiss_index(index, 'faiss_index.index')  # Save it as faiss_index.index

if __name__ == "__main__":
    create_and_save_index()
