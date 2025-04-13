"""
LangChain-based chatbot implementation for X-Customer Support.
Simplified version for improved reliability.
"""

import os
import logging
from typing import List, Optional, Union, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import standard libraries
from dotenv import load_dotenv
load_dotenv()

# Import langchain components
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
except ImportError:
    logger.error("Required packages not found. Install with: pip install langchain langchain-community")
    raise

# Import local modules
try:
    from src.retrieval import ContextRetriever
except ImportError:
    try:
        from retrieval import ContextRetriever
    except ImportError:
        logger.error("Could not import ContextRetriever")
        raise

class XSupportChatbot:
    """Twitter/X Support Chatbot using LangChain and RAG."""
    
    # Detailed prompt templates for more comprehensive answers
    PROMPT_TEMPLATES = {
        "standard": """
You are a Twitter/X customer support assistant providing detailed and helpful responses. You should always provide step-by-step instructions when appropriate, include security recommendations, and be thorough in your explanations.

When providing guidance on:
- Account security: Include specific steps for enabling additional security measures
- Password changes: Provide complete process with link instructions
- Account recovery: Offer detailed steps for various recovery scenarios
- Account deletion: Explain both deactivation and permanent deletion options and their differences
- Handle changes: Explain limitations, waiting periods, and potential complications

Context information (use this to inform your answer): 
{context}

User Query: {query}

Previous Conversation: {chat_history}

Provide a comprehensive and detailed answer that completely addresses the user's question:
""",
        "expert": """
You are a senior Twitter/X support specialist with extensive platform knowledge. Your responses should be detailed, accurate, and comprehensive, always including:

1. Step-by-step instructions with clear navigation paths through the Twitter/X interface
2. Security best practices and recommendations 
3. Potential complications or limitations users might encounter
4. Alternative approaches when available
5. Timeline expectations where applicable

For account security issues:
- Provide detailed steps for two-factor authentication setup
- Explain connected device management
- Describe suspicious activity indicators
- Include recovery options

For account changes:
- Detail all settings paths with exact menu locations
- Explain any waiting periods or verification steps
- Note any potential impacts to followers, verification, or connected services

Context information (reference this in your answer): 
{context}

User Query: {query}

Previous Conversation: {chat_history}

Provide a comprehensive expert response with detailed steps and recommendations:
""",
        "concise": """
You are a Twitter/X support assistant providing clear, structured answers. While being concise, you must include all necessary details and steps.

Your response should:
- Start with a direct answer to the query
- Include numbered steps when instructions are needed
- Mention important caveats or limitations
- Provide specific menu paths (e.g., "Settings → Account → Username")
- Include security recommendations when relevant

Context information: 
{context}

User Query: {query}

Previous Conversation: {chat_history}

Provide a clear, structured answer with all necessary details:
"""
    }
    
    def __init__(
        self,
        retriever: Optional[ContextRetriever] = None,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        prompt_template: str = "standard",
        use_memory: bool = True,
        memory_k: int = 3,
        top_k: int = 3,
        index_path: Optional[str] = None,
        data_path: Optional[str] = None,
    ):
        """Initialize the chatbot with the given parameters."""
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Create retriever if not provided
        if retriever:
            self.retriever = retriever
        else:
            if not index_path or not data_path:
                raise ValueError("If retriever is not provided, index_path and data_path are required")
            
            logger.info(f"Initializing ContextRetriever")
            self.retriever = ContextRetriever(
                index_path=index_path,
                data_path=data_path
            )
        
        # Store parameters
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.use_memory = use_memory
        self.memory_k = memory_k
        
        # Set prompt template
        if prompt_template in self.PROMPT_TEMPLATES:
            template = self.PROMPT_TEMPLATES[prompt_template]
        else:
            template = self.PROMPT_TEMPLATES["standard"]
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query", "chat_history"]
        )
        
        # Initialize memory
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="query",
                return_messages=True,
                k=memory_k
            )
        else:
            self.memory = None
        
        # Initialize conversation history for when memory is not used
        self.conversation_history = []
        
    def process_query(self, query: str) -> str:
        """Process a user query and generate a response."""
        try:
            # Get context
            contexts = self.retriever.retrieve(query, top_k=self.top_k)
            context_text = "\n\n".join(contexts) if contexts else "No specific information available."
            
            # Create LLM
            llm = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                temperature=self.temperature
            )
            
            # Create chain
            if self.use_memory:
                chain = LLMChain(
                    llm=llm,
                    prompt=self.prompt,
                    memory=self.memory,
                    verbose=False
                )
                
                # Run chain with memory
                response = chain({"context": context_text, "query": query})
                return response["text"]
            else:
                # No memory case
                chain = LLMChain(
                    llm=llm,
                    prompt=self.prompt,
                    verbose=False
                )
                
                # Prepare chat history
                chat_history = "\n".join([f"{item['role']}: {item['content']}" 
                                         for item in self.conversation_history[-6:]])
                
                # Run chain without memory
                response = chain({"context": context_text, "query": query, "chat_history": chat_history})
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": response["text"]})
                
                return response["text"]
                
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"I encountered an error. Please try again."
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.use_memory:
            self.memory.clear()
        self.conversation_history = []