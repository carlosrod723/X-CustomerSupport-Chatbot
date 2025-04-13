"""
Context retrieval module for X-Customer Support Chatbot.

This module handles the retrieval of relevant context from the FAISS index
using semantic similarity search based on user queries.
"""

import os
import time
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for available hardware acceleration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class ContextRetriever:
    """Class for retrieving relevant context using FAISS index and sentence embeddings."""
    
    def __init__(
        self,
        index_path: str,
        data_path: str,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: Optional[str] = None,
        text_column: str = 'cleaned_text',
        nprobe: int = 20  # Higher value for more accurate search
    ):
        """
        Initialize the ContextRetriever with necessary components.
        
        Args:
            index_path: Path to the FAISS index file
            data_path: Path to the data file (CSV or numpy embeddings)
            model_name: Name of the SentenceTransformer model
            device: Device to use for model inference ('cuda', 'mps', or 'cpu')
            text_column: Column name containing the text data
            nprobe: Number of cells to visit for IVF index (higher = more accurate)
        """
        self.text_column = text_column
        self.device = device if device else DEVICE
        
        # Load the embeddings model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Load the FAISS index
        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Configure index parameters if applicable (for IVF indexes)
        if isinstance(self.index, faiss.IndexIVF):
            logger.info(f"Setting nprobe to {nprobe} for IVF index")
            self.index.nprobe = nprobe
        
        # Load the data
        logger.info(f"Loading data from: {data_path}")
        self._load_data(data_path)
        
        logger.info("Context retriever initialized successfully")
    
    def _load_data(self, data_path: str):
        """
        Load data based on file extension.
        
        Args:
            data_path: Path to the data file
        """
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            self.data = pd.read_csv(data_path)
            
            # Check if the text column exists
            if self.text_column not in self.data.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in data")
            
            # Check if there's a separate numpy file with embeddings
            np_path = os.path.splitext(data_path)[0] + '.npy'
            if os.path.exists(np_path):
                logger.info(f"Loading embeddings from: {np_path}")
                self.embeddings = np.load(np_path)
            else:
                # If embeddings are stored in the CSV
                if 'embedding' in self.data.columns:
                    logger.info("Converting stored embeddings to numpy array")
                    self.embeddings = np.array(self.data['embedding'].tolist())
        
        elif file_ext == '.npy':
            # Assuming this is just the embeddings
            self.embeddings = np.load(data_path)
            
            # Try to load corresponding CSV with the same base name
            csv_path = os.path.splitext(data_path)[0] + '.csv'
            if os.path.exists(csv_path):
                logger.info(f"Loading text data from: {csv_path}")
                self.data = pd.read_csv(csv_path)
            else:
                logger.warning("No corresponding CSV found for text data")
                # Create a dummy dataframe with indices
                self.data = pd.DataFrame({
                    'index': range(len(self.embeddings)),
                    self.text_column: [f"Text_{i}" for i in range(len(self.embeddings))]
                })
        
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        logger.info(f"Loaded data with {len(self.data)} rows")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        return_distances: bool = False,
        reranking: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Retrieve relevant context for a given query.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            return_distances: Whether to return similarity scores
            reranking: Whether to apply a simple reranking strategy
            
        Returns:
            Either a list of context strings, or a tuple of (contexts, distances)
        """
        try:
            # Encode the query
            start_time = time.time()
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search the FAISS index
            distances, indices = self.index.search(query_embedding, top_k if not reranking else top_k * 2)
            
            # Get the corresponding text from indices
            results = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.data):
                    continue  # Skip invalid indices
                
                text = self.data.iloc[idx][self.text_column]
                distance = distances[0][i]
                
                results.append(text)
                result_distances.append(distance)
            
            # Simple reranking if enabled (retrieve more results then rerank)
            if reranking and len(results) > top_k:
                # This is a simple reranking - for a real system, you might use a more
                # sophisticated approach like a cross-encoder model
                reranked = sorted(zip(results, result_distances), key=lambda x: x[1])
                results = [r[0] for r in reranked[:top_k]]
                result_distances = [r[1] for r in reranked[:top_k]]
            
            elapsed = time.time() - start_time
            logger.debug(f"Retrieved {len(results)} contexts in {elapsed:.4f} seconds")
            
            if return_distances:
                return results, result_distances
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            if return_distances:
                return [], []
            else:
                return []
    
    def get_document_by_id(self, doc_id: int) -> str:
        """
        Get a document by its index ID.
        
        Args:
            doc_id: The document index
            
        Returns:
            The text of the document
        """
        if 0 <= doc_id < len(self.data):
            return self.data.iloc[doc_id][self.text_column]
        else:
            return ""
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: int = 3
    ) -> List[List[str]]:
        """
        Batch retrieve contexts for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of top results to return per query
            
        Returns:
            List of context lists for each query
        """
        # Encode all queries at once (more efficient)
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        
        # Search the FAISS index for all queries
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # Get the corresponding text from indices for each query
        all_results = []
        for i, query_indices in enumerate(indices):
            results = []
            for idx in query_indices:
                if idx >= 0 and idx < len(self.data):  # Check for valid index
                    results.append(self.data.iloc[idx][self.text_column])
            all_results.append(results)
        
        return all_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Retrieve context using FAISS index.')
    
    parser.add_argument(
        '--index', '-i',
        type=str,
        default='models/faiss_index_flat.index',
        help='Path to the FAISS index file'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='models/data_with_embeddings_ref.csv',
        help='Path to the data file (CSV or numpy)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='SentenceTransformer model to use'
    )
    
    parser.add_argument(
        '--text-column', '-t',
        type=str,
        default='cleaned_text',
        help='Column name containing the text data'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        default='How do I reset my Twitter password?',
        help='Query to test the retrieval'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Number of top results to return'
    )
    
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create retriever
    retriever = ContextRetriever(
        index_path=args.index,
        data_path=args.data,
        model_name=args.model,
        text_column=args.text_column
    )
    
    # Test retrieval
    logger.info(f"Testing retrieval with query: '{args.query}'")
    results, distances = retriever.retrieve(args.query, top_k=args.top_k, return_distances=True)
    
    # Print results
    logger.info(f"Retrieved {len(results)} contexts:")
    for i, (text, distance) in enumerate(zip(results, distances)):
        logger.info(f"{i+1}. [Score: {1/(1+distance):.4f}] {text[:100]}...")