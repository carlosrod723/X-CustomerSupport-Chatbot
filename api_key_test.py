"""
OpenAI API test using direct requests with proper dotenv loading.
This bypasses potential issues with the OpenAI Python package.
"""

import os
import sys
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    print("Loading environment variables from .env file...")
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Continuing without loading .env file...")

# Get API key from command line, environment, or .env (loaded above)
api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('OPENAI_API_KEY')

if not api_key:
    print("Error: No API key found. Check your .env file contains OPENAI_API_KEY=your-key-here")
    sys.exit(1)

print(f"Testing API key: {api_key[:5]}...{api_key[-4:]}")

# OpenAI API endpoint
url = "https://api.openai.com/v1/chat/completions"

# Request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Request payload
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 20
}

try:
    # Make the API request directly
    print("Making request to OpenAI API...")
    response = requests.post(url, headers=headers, json=payload)
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"✅ Success! API key is working.")
        print(f"Response: {content}")
        
        # Test GPT-4 access
        print("\nTesting GPT-4 access...")
        gpt4_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Confirm you are GPT-4"}],
            "max_tokens": 20
        }
        
        gpt4_response = requests.post(url, headers=headers, json=gpt4_payload)
        
        if gpt4_response.status_code == 200:
            gpt4_content = gpt4_response.json()["choices"][0]["message"]["content"]
            print(f"✅ GPT-4 access confirmed!")
            print(f"Response: {gpt4_content}")
        else:
            print(f"❌ GPT-4 access failed with status code {gpt4_response.status_code}")
            print(f"Error: {gpt4_response.text}")
    else:
        print(f"❌ Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 401:
            print("\nAPI key appears to be invalid or has expired.")
        elif response.status_code == 429:
            print("\nRate limit exceeded or insufficient quota.")
            
except Exception as e:
    print(f"❌ Error making request: {str(e)}")
    
print("\nIf you're having issues with the OpenAI package in your main code, try:")
print("1. pip uninstall openai")
print("2. pip install openai --upgrade")
print("3. Check for any proxy settings in your environment")