# Import necessary libraries and packages
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the .env file
openai_api_key= os.getenv('OPENAI_API_KEY')

# Function to generate response using GPT-3.5-turbo
def generate_response(query, context):
    response= openai.chat.completions.create(
        model= 'gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are a helpful customer support chatbot for Twitter/X.'},
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': context},
        ],
        max_tokens= 200
    )
    
    # Accessing the response text using dot notation
    return response.choices[0].message.content

# Example usage
query= 'How do I reset my Twitter password?'
context= "You can reset your Twitter password by going to the login page and selecting 'Forgot Password'."
response= generate_response(query, context)
print(f'Chatbot response: {response}')