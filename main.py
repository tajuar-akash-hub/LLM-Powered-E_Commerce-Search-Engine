import streamlit as st
import faiss
import numpy as np
from embedding import generate_bert_embedding
from faiss_search import load_product_data  # Import the function to load product data

# Load FAISS index
def load_faiss_index():
    try:
        index = faiss.read_index('faiss_index.index')  # Ensure this is the correct path
        return index
    except Exception as e:
        st.error(f"Failed to load the FAISS index: {e}")
        return None

# Search FAISS index
def search_embeddings_with_faiss(query_embedding, index, top_k=5):
    try:
        # Perform the search
        distances, indices = index.search(query_embedding, top_k)
        return distances, indices
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return None, None

# Main Streamlit app
def run_app():
    # Load FAISS index once (when the app is loaded)
    index = load_faiss_index()
    if index is None:
        return  # If the index could not be loaded, stop execution

    st.title("Product Search")

    # User input for search query
    query = st.text_input("Enter a query to find products:")

    # Add buttons for Search and Clear
    search_button = st.button("Search")
    clear_button = st.button("Clear Search")

    if clear_button:
        # Clear the search input box
        query = ""

    if search_button and query:
        # Generate the embedding for the query
        query_embedding = generate_bert_embedding(query).reshape(1, -1).astype(np.float32)  # Reshaping for FAISS

        # Search the FAISS index
        distances, indices = search_embeddings_with_faiss(query_embedding, index, top_k=9)  # Get 9 products

        # Handle no results found
        if indices is None or len(indices) == 0:
            st.write("No results found.")
            return

        # Load product data (from JSON)
        products = load_product_data('products.json')

        # Display search results inside cards
        st.subheader("Search Results:")

        # Create a grid of 3 columns for displaying 3 cards per row
        columns = st.columns(3)

        for i in range(len(indices[0])):
            product = products[indices[0][i]]  # Assuming products are a list of dictionaries
            score = distances[0][i]

            # Create the product card layout
            product_card = f"""
            <div style="background-color: #f4f4f4; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <h3 style="color: #2e3d49;">{product['ProductTitle']}</h3>
                <p><strong>Category:</strong> {product['Category']}</p>
                <p><strong>SubCategory:</strong> {product['SubCategory']}</p>
                <p><strong>Product Type:</strong> {product['ProductType']}</p>
                <p><strong>Colour:</strong> {product['Colour']}</p>
                <p><strong>Usage:</strong> {product['Usage']}</p>
                <p><strong>Product Id:</strong> {product['ProductId']}</p>
                <p><strong>Score:</strong> {score:.4f}</p>
                <p><strong>Image:</strong> <img src="{product['ImageURL']}" style="max-width: 200px; border-radius: 10px;" /></p>
            </div>
            """

            # Display each product card in the corresponding column
            col_idx = i % 3  # Get the column index (0, 1, or 2)
            columns[col_idx].markdown(product_card, unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()
