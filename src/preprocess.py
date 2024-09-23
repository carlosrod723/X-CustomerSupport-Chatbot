# Import necessary libraries and packages
import re
import spacy
import pandas as pd
from contractions import fix  

# Load spaCy's English tokenizer
nlp= spacy.load('en_core_web_sm')

# Function to tokenize using spaCy
def preprocess_text(text):

    # Remove URLs
    text= re.sub(r'http\S+|www\S+|https\S+', '', text, flags= re.MULTILINE)
    
    # Remove or replace special characters and emojis
    text= re.sub(r'[^A-Za-z0-9\s]', '', text)  
    text= text.encode('ascii', 'ignore').decode('ascii')  
    
    # Convert text to lowercase
    text= text.lower()
    
    # Expand contractions (e.g., "don't" -> "do not")
    text= fix(text)
    
    # Remove Twitter-specific elements (mentions and hashtags)
    text= re.sub(r'@\w+', '', text)  
    text= re.sub(r'#\w+', '', text) 
    
    # Tokenize text using spaCy
    doc= nlp(text)
    tokens= [token.text for token in doc]
    
    # Remove stopwords using spaCy's built-in stopwords
    tokens= [word for word in tokens if word not in spacy.lang.en.stop_words.STOP_WORDS]
    
    # Normalize text (correct spelling, standardize elongated words)
    tokens= [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]  
    
    # Handle negations (basic handling by keeping the word "not" with the next word)
    clean_tokens= []
    i= 0
    while i < len(tokens):
        if tokens[i] == 'not' and i + 1 < len(tokens):
            clean_tokens.append(f'not_{tokens[i+1]}')
            i += 2
        else:
            clean_tokens.append(tokens[i])
            i += 1

    return ' '.join(clean_tokens)

# Load the dataset
data_path= '/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/Data/tweets.csv'
data= pd.read_csv(data_path)

# Check for missing values in the 'text' column and drop them if any
missing_text_count= data['text'].isnull().sum()
print(f'Number of missing entries in "text" column: {missing_text_count}')

# Drop rows where 'text' is missing
data= data.dropna(subset=['text'])

# Apply the preprocessing function to the dataset
data['cleaned_text']= data['text'].apply(preprocess_text)

# Save the cleaned dataset to a new CSV file
cleaned_data_path= '/Users/lidiaacosta/Desktop/Carlos_Projects/MyProjects/X-CustomerSupport-Chatbot/Data/tweets_cleaned.csv'
data.to_csv(cleaned_data_path, index=False)

print(f'Cleaned dataset saved to {cleaned_data_path}')