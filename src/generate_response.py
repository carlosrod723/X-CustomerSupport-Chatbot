"""
Response generation module for X-Customer Support Chatbot.

This module handles the generation of responses using OpenAI's language models
based on user queries and retrieved context.
"""

import os
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import environment variable handling
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    logger.error("python-dotenv not found. Please install it with: pip install python-dotenv")
    raise

# Load environment variables
load_dotenv()

# Import OpenAI with compatibility handling
try:
    # Try importing the new client
    from openai import OpenAI
    HAS_NEW_CLIENT = True
    logger.info("Using OpenAI Python v1.x client")
except (ImportError, AttributeError):
    # Fall back to old client
    import openai
    HAS_NEW_CLIENT = False
    logger.info("Using OpenAI Python v0.x client")

class ResponseGenerator:
    """Class for generating responses using OpenAI's language models."""
    
    # System prompt templates for different scenarios
    SYSTEM_PROMPTS = {
        "standard": """You are a helpful customer support assistant for Twitter/X. 
Your goal is to provide clear and accurate information to users about Twitter/X products and services.
Use a professional, friendly tone. Be concise but thorough.""",
        
        "technical": """You are a technical support specialist for Twitter/X.
Provide detailed, step-by-step instructions to solve technical issues.
Focus on accuracy and clarity. Use technical terms when appropriate but explain them clearly.""",
        
        "security": """You are a security specialist for Twitter/X.
Prioritize account security and data protection in your responses.
Be thorough and emphasize best practices for maintaining account security.""",
        
        "concise": """You are a customer support chatbot for Twitter/X.
Provide extremely brief, direct answers to user questions.
Focus only on essential information and actionable steps."""
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt_type: str = "standard"
    ):
        """
        Initialize the response generator.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY from environment)
            model: Language model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in the generated response
            system_prompt_type: Type of system prompt to use
        """
        # Load environment variables and get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Store parameters
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up system prompt
        if system_prompt_type in self.SYSTEM_PROMPTS:
            self.system_prompt = self.SYSTEM_PROMPTS[system_prompt_type]
        else:
            logger.warning(f"Unknown prompt type: {system_prompt_type}. Using standard.")
            self.system_prompt = self.SYSTEM_PROMPTS["standard"]
        
        # Initialize OpenAI client
        if HAS_NEW_CLIENT:
            # Initialize with minimal parameters to avoid errors
            self.client = OpenAI(api_key=self.api_key)
        else:
            # For older client versions
            openai.api_key = self.api_key
        
        logger.info(f"Response generator initialized with model: {model}")
    
    def _format_context(self, context: Union[str, List[str]]) -> str:
        """
        Format context into a string suitable for the prompt.
        
        Args:
            context: Context string or list of context strings
            
        Returns:
            Formatted context string
        """
        if isinstance(context, list):
            if not context:
                return "No additional context available."
            
            # Join multiple context items with clear separation
            formatted = "Here is some relevant information that might help:\n\n"
            for i, ctx in enumerate(context, 1):
                formatted += f"Context {i}: {ctx}\n\n"
            return formatted
        else:
            return context
    
    def generate(
        self,
        query: str,
        context: Union[str, List[str]] = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        callback=None
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and context.
        
        Args:
            query: User query
            context: Context information to inform the response
            conversation_history: Previous messages in the conversation
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream the response
            callback: Callback function for streaming responses
            
        Returns:
            Dictionary containing response text and metadata
        """
        # Start timing
        start_time = time.time()
        
        # Format context if it's a list
        formatted_context = self._format_context(context)
        
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add context and query
        if formatted_context:
            messages.append({"role": "user", "content": f"Please use this information to help answer the user's question: {formatted_context}"})
            messages.append({"role": "assistant", "content": "I'll use this information to help answer your question."})
        
        messages.append({"role": "user", "content": query})
        
        # Use specified values or defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            logger.info(f"Generating response with {len(messages)} messages")
            
            if HAS_NEW_CLIENT:
                # New OpenAI client
                if stream:
                    # Handle streaming
                    stream_resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        stream=True
                    )
                    
                    # Process streaming response
                    full_response = ""
                    for chunk in stream_resp:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                if callback:
                                    callback(content)
                    
                    # Create response dict similar to non-streaming
                    response_data = {
                        "text": full_response,
                        "model": self.model,
                        "usage": {"total_tokens": "unknown"},
                        "elapsed_time": time.time() - start_time
                    }
                
                else:
                    # Regular non-streaming request
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens
                    )
                    
                    # Extract response data
                    response_data = {
                        "text": response.choices[0].message.content,
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        },
                        "elapsed_time": time.time() - start_time
                    }
            
            else:
                # Legacy OpenAI client
                if stream:
                    # Handle streaming
                    stream_resp = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        stream=True
                    )
                    
                    full_response = ""
                    for chunk in stream_resp:
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                content = chunk['choices'][0]['delta']['content']
                                if content:
                                    full_response += content
                                    if callback:
                                        callback(content)
                    
                    response_data = {
                        "text": full_response,
                        "model": self.model,
                        "usage": {"total_tokens": "unknown"},
                        "elapsed_time": time.time() - start_time
                    }
                
                else:
                    # Regular non-streaming request
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens
                    )
                    
                    response_data = {
                        "text": response['choices'][0]['message']['content'],
                        "model": response['model'],
                        "usage": response['usage'],
                        "elapsed_time": time.time() - start_time
                    }
            
            # Log completion statistics
            logger.info(
                f"Response generated in {response_data['elapsed_time']:.2f}s, "
                f"Model: {response_data['model']}"
            )
            
            return response_data
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "text": f"I'm sorry, I encountered an error while processing your request. Error: {str(e)}",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def generate_with_retry(
        self,
        query: str,
        context: Union[str, List[str]] = "",
        max_retries: int = 3,
        retry_delay: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response with automatic retry on failure.
        
        Args:
            query: User query
            context: Context information
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional arguments to pass to generate()
            
        Returns:
            Response data dictionary
        """
        retries = 0
        while retries <= max_retries:
            try:
                response = self.generate(query, context, **kwargs)
                if "error" not in response:
                    return response
                
                # If there was an error, check if we should retry
                if "rate limit" in response["error"].lower() or "timeout" in response["error"].lower():
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * retries
                        logger.warning(f"Retry {retries}/{max_retries} after {wait_time}s delay. Error: {response['error']}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded. Last error: {response['error']}")
                        return response
                else:
                    # Don't retry for non-transient errors
                    return response
            
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    wait_time = retry_delay * retries
                    logger.warning(f"Exception occurred, retry {retries}/{max_retries} after {wait_time}s delay. Error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded after exception. Last error: {str(e)}")
                    return {
                        "text": f"I'm sorry, I encountered a persistent error while processing your request. Error: {str(e)}",
                        "error": str(e),
                        "elapsed_time": 0
                    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate responses for X-Customer Support Chatbot queries.')
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        default='How do I reset my Twitter password?',
        help='User query to test response generation'
    )
    
    parser.add_argument(
        '--context', '-c',
        type=str,
        default="You can reset your Twitter password by going to the login page and selecting 'Forgot Password'.",
        help='Context information to inform the response'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt-4',
        help='OpenAI model to use'
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='Temperature for response generation (0-1)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens in the generated response'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        help='OpenAI API key (if not provided, will use environment variable)'
    )
    
    parser.add_argument(
        '--system-prompt',
        type=str,
        choices=['standard', 'technical', 'security', 'concise'],
        default='standard',
        help='Type of system prompt to use'
    )
    
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream the response token by token'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create response generator
    generator = ResponseGenerator(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt_type=args.system_prompt
    )
    
    # Define callback for streaming
    def print_chunk(chunk):
        print(chunk, end='', flush=True)
    
    # Generate response
    if args.stream:
        print("Generating streaming response...")
        response_data = generator.generate(
            query=args.query,
            context=args.context,
            stream=True,
            callback=print_chunk
        )
        print("\n\nGeneration complete!")
        print(f"Model: {response_data['model']}")
        print(f"Time: {response_data['elapsed_time']:.2f}s")
    else:
        response_data = generator.generate(
            query=args.query,
            context=args.context
        )
        print(f"\nQuery: {args.query}")
        print(f"Context: {args.context}")
        print(f"Response: {response_data['text']}")
        print(f"Model: {response_data['model']}")
        if 'usage' in response_data and isinstance(response_data['usage'], dict):
            if 'total_tokens' in response_data['usage']:
                print(f"Total tokens: {response_data['usage']['total_tokens']}")
        print(f"Time: {response_data['elapsed_time']:.2f}s")