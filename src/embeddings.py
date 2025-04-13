"""
Embeddings generation module for X-Customer Support Chatbot.

This module generates and stores embeddings for preprocessed text data,
using SentenceTransformer models and FAISS for efficient vector storage and retrieval.
"""

import os
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for available hardware acceleration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_batch_size(model_name: str) -> int:
    """
    Determine appropriate batch size based on model and available memory.
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        int: Recommended batch size
    """
    # With 128GB RAM on M4 Max, we can use larger batch sizes
    if 'large' in model_name.lower():
        return 512
    elif 'base' in model_name.lower():
        return 1024
    else:
        return 2048  # For smaller models like MiniLM

def generate_embeddings(
    data: pd.DataFrame,
    text_column: str = 'cleaned_text',
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: Optional[int] = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for text data using a SentenceTransformer model.
    
    Args:
        data: DataFrame containing the text data
        text_column: Column name containing the preprocessed text
        model_name: Name of the SentenceTransformer model to use
        batch_size: Size of batches for processing (auto-determined if None)
        show_progress: Whether to show a progress bar
        
    Returns:
        np.ndarray: Generated embeddings matrix
    """
    # Ensure all values in text column are strings
    data[text_column] = data[text_column].fillna('')
    data[text_column] = data[text_column].astype(str)
    
    # Initialize the SentenceTransformer model with hardware acceleration
    logger.info(f"Initializing SentenceTransformer model: {model_name}")
    logger.info(f"Using device: {DEVICE}")
    start_time = time.time()
    
    model = SentenceTransformer(model_name, device=DEVICE)
    
    # Determine optimal batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size(model_name)
    
    logger.info(f"Using batch size: {batch_size}")
    
    # Generate embeddings with batching and progress bar
    texts = data[text_column].tolist()
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    
    if show_progress:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=DEVICE
        )
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            device=DEVICE
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Embedding generation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    return embeddings

def create_faiss_index(
    embeddings: np.ndarray,
    index_type: str = 'flat'
) -> faiss.Index:
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: Matrix of embeddings
        index_type: Type of FAISS index to create
        
    Returns:
        faiss.Index: The created FAISS index
    """
    # Get dimensionality of embeddings
    d = embeddings.shape[1]
    
    logger.info(f"Creating FAISS index with dimensionality {d}")
    
    # Create index based on type
    if index_type == 'flat':
        # Simple flat index - exact but slower for large datasets
        index = faiss.IndexFlatL2(d)
    elif index_type == 'ivf':
        # IVF index - faster but approximate
        # For your 128GB RAM system, we can use more Voronoi cells
        n_cells = min(4096, int(embeddings.shape[0] / 39))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_L2)
        # Need to train the index with the data
        logger.info(f"Training IVF index with {n_cells} cells")
        index.train(embeddings)
    elif index_type == 'hnsw':
        # HNSW index - very fast and accurate, but can cause memory issues
        # Using more conservative parameters
        m = 32  # Lower M value to reduce memory usage
        ef_construction = 100  # Lower value for better stability
        index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_L2)
        index.hnsw.efConstruction = ef_construction
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # For non-GPU implementation
    logger.info(f"Adding {embeddings.shape[0]} vectors to index")
    index.add(embeddings)
    
    return index

def process_embeddings(
    input_path: str,
    output_dir: str,
    text_column: str = 'cleaned_text',
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: Optional[int] = None,
    index_type: str = 'hnsw',  # HNSW is faster for this hardware
    save_embeddings_csv: bool = True
) -> Tuple[np.ndarray, faiss.Index]:
    """
    Main function to process text data, generate embeddings, and create FAISS index.
    
    Args:
        input_path: Path to the cleaned CSV data
        output_dir: Directory to save outputs
        text_column: Column containing the text to embed
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding generation
        index_type: Type of FAISS index to create
        save_embeddings_csv: Whether to save embeddings to CSV
        
    Returns:
        Tuple containing embeddings and FAISS index
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the cleaned dataset
        logger.info(f"Loading dataset from {input_path}")
        data = pd.read_csv(input_path)
        
        # Generate embeddings
        embeddings = generate_embeddings(
            data,
            text_column=text_column,
            model_name=model_name,
            batch_size=batch_size
        )
        
        # Create FAISS index
        index = create_faiss_index(embeddings, index_type=index_type)
        
        # Save the FAISS index with better error handling
        index_path = os.path.join(output_dir, f'faiss_index_{index_type}.index')
        logger.info(f"Saving FAISS index to {index_path}")
        try:
            # Write with explicit synchronization
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index successfully saved to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            logger.info("Trying to save with a simpler index type as fallback")
            # If the index saving fails, try with a simpler index as fallback
            if index_type != 'flat':
                fallback_index = faiss.IndexFlatL2(embeddings.shape[1])
                fallback_index.add(embeddings)
                fallback_path = os.path.join(output_dir, 'faiss_index_flat_fallback.index')
                faiss.write_index(fallback_index, fallback_path)
                logger.info(f"Fallback FAISS index saved to {fallback_path}")
            else:
                raise
        
        # Optionally save the embeddings in a CSV file
        if save_embeddings_csv:
            # We won't try to store embeddings in the dataframe anymore
            logger.info("Preparing to save embeddings data")
            
            # Save CSV with reference data only (no embeddings)
            embeddings_path = os.path.join(output_dir, 'data_with_embeddings_ref.csv')
            logger.info(f"Saving data with embeddings reference to {embeddings_path}")
            
            # Save without the actual embeddings to conserve space
            # Using 'textID' column instead of 'id', based on the actual data structure
            data[['textID', text_column]].to_csv(embeddings_path, index=False)
            
            # Save raw numpy embeddings for faster loading
            np_path = os.path.join(output_dir, 'embeddings.npy')
            logger.info(f"Saving raw embeddings to {np_path}")
            np.save(np_path, embeddings)
        
        logger.info("Embeddings processing completed successfully")
        return embeddings, index
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings for X-Customer Support text data.')
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/tweets_cleaned.csv',
        help='Path to the cleaned CSV file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models',
        help='Directory to save the embeddings and FAISS index'
    )
    
    parser.add_argument(
        '--text-column', '-t',
        type=str,
        default='cleaned_text',
        help='Column name containing the preprocessed text'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='SentenceTransformer model to use'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Batch size for embedding generation (default: auto)'
    )
    
    parser.add_argument(
        '--index-type',
        type=str,
        choices=['flat', 'ivf', 'hnsw'],
        default='flat',  # Changed default to 'flat' which is simpler and more stable
        help='Type of FAISS index to create'
    )
    
    parser.add_argument(
        '--no-save-csv',
        action='store_true',
        help='Do not save embeddings in CSV (saves memory and disk space)'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Process embeddings
    process_embeddings(
        input_path=args.input,
        output_dir=args.output_dir,
        text_column=args.text_column,
        model_name=args.model,
        batch_size=args.batch_size,
        index_type=args.index_type,
        save_embeddings_csv=not args.no_save_csv
    )