# Import necessary libraries and packages
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and embeddings
index= faiss.read_index('/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/faiss_index.index')
data= np.load('/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/cleaned_data_with_embeddings.csv')

# Initialize the Sentence Transformer model for encoding new queries
model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to retrieve relevant context from FAISS
def retrieve_context(query, top_k= 3):
    query_embedding= model.encode([query], convert_to_numpy= True)
    distances, indices= index.search(query_embedding, top_k)
    
    # Get the corresponding cleaned text from indices
    results= [data[i]['cleaned_text'] for i in indices[0]]
    return results

# Example usage:
query= 'How do I reset my Twitter password?'
context= retrieve_context(query)
print(f'Retrieved context: {context}')
