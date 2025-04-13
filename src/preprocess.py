"""
Text preprocessing module for X-Customer Support Chatbot.

This module handles the preprocessing of Twitter/X support text data,
including cleaning, normalization, and tokenization steps.
"""

import re
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import spacy
import pandas as pd
from tqdm import tqdm
from contractions import fix
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy's English tokenizer (with error handling)
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("Successfully loaded spaCy's English tokenizer")
except OSError:
    logger.error("Could not load spaCy's English model. You may need to download it first.")
    logger.info("Try running: python -m spacy download en_core_web_sm")
    raise

def preprocess_text(text: str) -> str:
    """
    Preprocess text data by removing URLs, special characters, 
    expanding contractions, and handling negations.
    
    Args:
        text (str): The input text to be preprocessed.
        
    Returns:
        str: The preprocessed text.
    """
    # Skip processing if text is None or empty
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove or replace special characters and emojis
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
        text = text.encode('ascii', 'ignore').decode('ascii')  
        
        # Convert text to lowercase
        text = text.lower()
        
        # Expand contractions (e.g., "don't" -> "do not")
        text = fix(text)
        
        # Remove Twitter-specific elements (mentions and hashtags)
        text = re.sub(r'@\w+', '', text)  
        text = re.sub(r'#\w+', '', text) 
        
        # Tokenize text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # Remove stopwords using spaCy's built-in stopwords
        tokens = [word for word in tokens if word not in spacy.lang.en.stop_words.STOP_WORDS]
        
        # Normalize text (correct spelling, standardize elongated words)
        tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]  
        
        # Improved negation handling
        clean_tokens = []
        i = 0
        negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'hardly', 'scarcely'}
        
        while i < len(tokens):
            if tokens[i] in negation_words and i + 1 < len(tokens):
                # Combine negation with the next token
                clean_tokens.append(f'not_{tokens[i+1]}')
                i += 2
            else:
                clean_tokens.append(tokens[i])
                i += 1
                
        return ' '.join(clean_tokens).strip()
    
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return ""

def process_chunk(chunk_data: Tuple[pd.DataFrame, str]) -> pd.DataFrame:
    """Process a chunk of the dataframe for multiprocessing."""
    chunk, text_column = chunk_data
    chunk['cleaned_text'] = chunk[text_column].apply(preprocess_text)
    return chunk

def preprocess_dataset(
    input_path: str, 
    output_path: str, 
    text_column: str = 'text',
    use_multiprocessing: bool = True,
    n_processes: Optional[int] = None
) -> None:
    """
    Load, preprocess, and save a dataset.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path where the cleaned CSV file will be saved.
        text_column (str): Column containing the text to process.
        use_multiprocessing (bool): Whether to use multiprocessing for faster processing.
        n_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Load the dataset
        logger.info(f"Loading dataset from {input_path}")
        data = pd.read_csv(input_path)
        
        # Check for missing values
        missing_text_count = data[text_column].isnull().sum()
        logger.info(f"Number of missing entries in '{text_column}' column: {missing_text_count}")
        
        # Drop rows where text is missing
        if missing_text_count > 0:
            data = data.dropna(subset=[text_column])
            logger.info(f"Dropped {missing_text_count} rows with missing text")
        
        logger.info(f"Processing {len(data)} rows of text data...")
        
        if use_multiprocessing and len(data) > 1000:  # Only use multiprocessing for larger datasets
            if n_processes is None:
                n_processes = max(1, cpu_count() - 1)  # Leave one CPU free
                
            logger.info(f"Using multiprocessing with {n_processes} processes")
            
            # Split data into chunks for processing
            chunk_size = max(1, len(data) // n_processes)
            chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            # Create a list of tuples with chunk and text_column
            chunk_data = [(chunk, text_column) for chunk in chunks]
            
            # Process chunks in parallel
            with Pool(processes=n_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_chunk, chunk_data),
                    total=len(chunks),
                    desc="Processing data chunks"
                ))
            
            # Combine processed chunks
            data = pd.concat(results, ignore_index=True)
        else:
            # Process sequentially with progress bar
            tqdm.pandas(desc="Preprocessing text")
            data['cleaned_text'] = data[text_column].progress_apply(preprocess_text)
        
        # Save the cleaned dataset
        logger.info(f"Saving cleaned dataset to {output_path}")
        data.to_csv(output_path, index=False)
        
        logger.info(f"Preprocessing complete. Cleaned dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess Twitter/X support data for the chatbot.')
    
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default='data/tweets.csv',
        help='Path to the input CSV file containing the raw data'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='data/tweets_cleaned.csv',
        help='Path where the cleaned CSV file will be saved'
    )
    
    parser.add_argument(
        '--text-column', '-t', 
        type=str, 
        default='text',
        help='Column name containing the text to process'
    )
    
    parser.add_argument(
        '--disable-multiprocessing', 
        action='store_true',
        help='Disable multiprocessing (use for debugging or on systems with limited resources)'
    )
    
    parser.add_argument(
        '--processes', '-p', 
        type=int, 
        default=None,
        help='Number of processes to use for multiprocessing (defaults to CPU count - 1)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run preprocessing
    preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        use_multiprocessing=not args.disable_multiprocessing,
        n_processes=args.processes
    )