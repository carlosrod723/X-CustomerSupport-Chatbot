#Import necessary libraries and packages
import os
import openai

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the .env file
openai_api_key= os.getenv('OPENAI_API_KEY')

try:

    # Test the key by making a simple request to the OpenAI API
    response= openai.ChatCompletion.create(
        model= 'gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello!'}
        ]
    )
    print('API Key is valid!')
    print(response['choices'][0]['message']['content'])

except openai.error.InvalidRequestError as e:
    print(f'There was an error with the request: {e}')
except openai.error.AuthenticationError as e:
    print(f'Invalid API key. Please check your key and try again. Error: {e}')
except Exception as e:
    print(f'An error occurred: {e}')