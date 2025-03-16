import json
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import faiss

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings using BERT
def generate_bert_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get BERT model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # We take the embeddings from the last hidden state (mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embeddings

# Load product data from JSON file and return only the first `num_items` products
def load_product_data(file_path):
    with open(file_path, 'r') as file:
        products = json.load(file)
    # Return only the first `num_items` products (or fewer if there are less than 100)
    return products


# def load_product_data(file_path, num_items=100):
#     with open(file_path, 'r') as file:
#         products = json.load(file)
#     # Return only the first `num_items` products (or fewer if there are less than 100)
#     return products[:num_items]

# Process the products to generate embeddings using available columns
def process_products(products):
    embeddings = []
    product_ids = []
    for product in products:
        # Use relevant fields for generating embeddings
        title = product.get('ProductTitle', '')
        gender = product.get('Gender', '')
        category = product.get('Category', '')
        subcategory = product.get('SubCategory', '')
        product_type = product.get('ProductType', '')
        colour = product.get('Colour', '')
        usage = product.get('Usage', '')

        # Combine all text fields into a description for embedding
        text = f"Product Title: {title}. Gender: {gender}. Category: {category}. SubCategory: {subcategory}. Product Type: {product_type}. Colour: {colour}. Usage: {usage}"
        
        # Generate embedding for the combined text
        emb = generate_bert_embedding(text)
        embeddings.append(emb)  # Store the embedding
        product_ids.append(product['ProductId'])  # Store product id
    
    return np.array(embeddings, dtype=np.float32), product_ids

# Save embeddings to FAISS index
def save_embeddings_to_faiss(embeddings, output_file_faiss):
    d = embeddings.shape[1]  # Dimensionality of embeddings (768 for BERT)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(d)
    
    # Add embeddings to FAISS index
    index.add(embeddings)
    
    # Save the FAISS index to a file
    faiss.write_index(index, output_file_faiss)
    print(f"FAISS index saved to {output_file_faiss}")

# Main function to orchestrate the process
if __name__ == "__main__":
    # Load products from JSON file
    products = load_product_data('products.json')  # Path to your products JSON file
    
    # Process products and generate embeddings
    embeddings, product_ids = process_products(products)
    
    # Save embeddings to FAISS index file
    save_embeddings_to_faiss(embeddings, 'faiss_index.index')  # Output file to save the FAISS index

    print("Embeddings have been saved to FAISS index successfully.")
