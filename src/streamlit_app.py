# Import necessary libraries and packages
import os
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the .env file
openai_api_key= os.getenv('OPENAI_API_KEY')

# Initialize OpenAI model using LangChain
llm= ChatOpenAI(model='gpt-3.5-turbo', openai_api_key= openai_api_key)

# Define ReACT-based prompt template
template = '''
You are a helpful customer support assistant for Twitter/X. Given the following context, answer the user's question in a clear and detailed manner:
Context: {context}
User Query: {query}
Assistant's Answer:
'''

prompt= PromptTemplate(template=template, input_variables=['context', 'query'])

# Load FAISS index and embeddings
index= faiss.read_index('/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/faiss_index.index')
data= pd.read_csv('/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/models/cleaned_data_with_embeddings.csv')


# SentenceTransformer model for generating query embeddings
embed_model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to retrieve relevant context from FAISS
def retrieve_context(query, top_k= 3):
    query_embedding= embed_model.encode([query], convert_to_numpy= True)
    distances, indices= index.search(query_embedding, top_k)
    
    # Get the corresponding cleaned text from indices
    results= [data.iloc[i]['cleaned_text'] for i in indices[0]]
    return ' '.join(results)

# Create the LangChain QA chain
def generate_response(query):
    context= retrieve_context(query)
    llm_chain= LLMChain(llm= llm, prompt= prompt)
    response= llm_chain.run({'context': context, 'query': query})
    return response

# Streamlit UI
st.title("Twitter/X Customer Support Chatbot")

# Input from user
user_query= st.text_input('Enter your query here:')

# Display chatbot response
if st.button('Submit'):
    if user_query:
        st.write('Generating response...')
        response= generate_response(user_query)
        st.write(f'Chatbot Response: {response}')
    else:
        st.write('Please enter a query.')
