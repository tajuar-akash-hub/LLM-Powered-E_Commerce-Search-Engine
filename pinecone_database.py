from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone instance with valid environment and API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")

# Define the name of the index (must be lowercase and can only contain alphanumeric characters and dashes)
index_name = 'my-index-1'  # Ensure this follows the correct format

# Check if the index already exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,  # Valid index name here
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(cloud='gcp', region='us-east1')
    )


# Connect to the Pinecone index
index = pc.Index(index_name)

def search_embeddings(query_embedding, top_k=3):
    """
    This function performs a similarity search in Pinecone using the query embedding.
    
    Args:
        query_embedding (list): The query's embedding.
        top_k (int): The number of top results to retrieve (default is 3).
        
    Returns:
        dict: The search results from Pinecone.
    """
    # Perform the search query in Pinecone
    query_results = index.query(
        vector=query_embedding,  # The query vector
        top_k=top_k,  # The number of closest matches to return
        include_metadata=True  # Whether to include metadata (like product details)
    )
    return query_results

# Example: Function to insert embeddings (this would typically be called after generating embeddings for products)
def insert_embeddings(embeddings):
    """
    Insert embeddings into the Pinecone index.
    
    Args:
        embeddings (list of tuples): A list of (id, embedding) tuples to insert.
    """
    index.upsert(vectors=embeddings)  # Insert embeddings into Pinecone

# Example usage:
# Save embeddings as a list of (id, embedding) tuples and insert into Pinecone
# embeddings = [(str(product['id']), generate_bert_embedding(product['expanded_description'])) for product in products]
# insert_embeddings(embeddings)
