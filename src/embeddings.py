# Import necessary libraries and packages
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load the cleaned dataset
data_path= '/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/Data/tweets_cleaned.csv'
data= pd.read_csv(data_path)

# Ensure all values in 'cleaned_text' are strings
data['cleaned_text']= data['cleaned_text'].fillna('')  
data['cleaned_text']= data['cleaned_text'].astype(str)  

# Initialize the SentenceTransformer model
model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for the cleaned text
print('Generating embeddings...')
embeddings= model.encode(data['cleaned_text'].tolist(), convert_to_numpy=True)

# Save the embeddings in FAISS index
index= faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, '/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/faiss_index.index')

# Save the embeddings along with the cleaned text in a new CSV file
data['embeddings']= [emb.tolist() for emb in embeddings]
data.to_csv('/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/cleaned_data_with_embeddings.csv', index= False)

print('Embeddings and FAISS index saved.')