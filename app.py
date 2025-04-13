"""
Streamlit web application for X-Customer Support Chatbot.
Simple version with minimal styling and dependencies.
"""

import streamlit as st
from src.chatbot_with_langchain import XSupportChatbot

# Page config
st.set_page_config(
    page_title="X Customer Support",
    page_icon="🐦",
    layout="wide"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = XSupportChatbot(
            model_name="gpt-4",  # Using more capable model for better answers
            prompt_template="expert",  # Using expert template for detailed responses
            temperature=0.7,
            top_k=3,  # Increased to get more context
            use_memory=True,
            index_path="models/faiss_index_flat.index",
            data_path="models/data_with_embeddings_ref.csv"
        )
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.stop()

if "generating" not in st.session_state:
    st.session_state.generating = False

# Import Google Fonts at the top and add critical CSS to hide helper text
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Basic typography improvements */
    body {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1E90FF;
    }
    
    /* Target all known helper text classes with direct CSS classes */
    .css-1vj29lo, .css-0, .css-f3zii7, .css-xgfwob, .css-l48j42, 
    .css-1j6vque, .css-1gksd2l, .css-19rxjzo, .css-19yh7wc {
        display: none !important;
    }
    
    /* Hide all small elements that might contain the helper text */
    small {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Add chat message styling
st.markdown("""
<style>
    /* Chat container */
    .chat-container {
        background-color: #EBFCFF;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(170, 241, 255, 0.5);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #1E90FF, #2BA3EC);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 18px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        font-family: 'Poppins', sans-serif;
        position: relative;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Bot message styling */
    .bot-message {
        background-color: white;
        color: #333;
        border-radius: 18px 18px 18px 0;
        padding: 12px 18px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        font-family: 'Poppins', sans-serif;
        border-left: 3px solid #2BA3EC;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    /* Fix for floats */
    .clearfix {
        clear: both;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Create a chat container if there are messages
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Add a clearfix to handle floating elements
    st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show welcome message as a title without container border
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; margin: 20px 0;">
        <img src="https://abs.twimg.com/responsive-web/client-web/icon-default.522d363a.png" width="60" style="margin-bottom: 15px;">
        <h1 style="color: #1E90FF; margin-bottom: 10px; font-size: 2.2rem;">Welcome to X Support</h1>
        <p style="color: #666; margin-bottom: 15px;">Ask any question about your Twitter/X account below.</p>
        <p style="color: #2BA3EC; font-size: 0.9rem;">See the sidebar for example questions →</p>
    </div>
    """, unsafe_allow_html=True)

# Process user input when submit button is clicked
def submit_clicked():
    # Get input from the text area instead of text input
    user_input = st.session_state.user_input_area.strip() if "user_input_area" in st.session_state else ""
    
    if not user_input:
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Get bot response
    try:
        response = st.session_state.chatbot.process_query(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Clear the input field after submission
    st.session_state.user_input_area = ""

# Style for the input area
st.markdown("""
<style>
    /* Completely eliminate all extra containers and gray areas */
    div.stTextInput > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div.stTextInput > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Style for textarea instead of text input */
    .stTextArea > div > div {
        border-radius: 30px !important;
        border: 2px solid #1E90FF !important; /* Dodger blue border */
        padding: 0 !important;
        margin: 0 !important;
        background: white !important;
        position: relative !important;
    }
    
    .stTextArea textarea {
        border-radius: 30px !important;
        padding: 10px 35px 10px 20px !important; /* Added extra right padding for character count */
        font-family: 'Poppins', sans-serif !important;
        color: #333 !important;
        background: white !important;
        width: 100% !important;
        height: 44px !important;
        min-height: 44px !important;
        max-height: 44px !important;
        resize: none !important;
        overflow: hidden !important;
        line-height: 24px !important;
        border: none !important;
    }
    
    /* Position the character count better */
    .stTextArea div[data-testid="stTextAreaCharCounter"] {
        position: absolute !important;
        right: 15px !important; /* Move away from border */
        bottom: 10px !important;
        font-size: 12px !important;
        color: #6e6e6e !important;
        z-index: 1 !important;
    }
    
    .stTextArea textarea:focus {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Extremely aggressive targeting to remove ALL helper text, hints and labels */
    /* Hide all small text elements anywhere in inputs */
    .stTextInput small, 
    .stTextInput [data-testid="stText"] small,
    div[data-testid="stFormHelperText"],
    [class*="Hint"], 
    [data-baseweb*="hint"],
    [data-baseweb*="help"],
    .stTextInput span[class*="caption"],
    div[role="textbox"] ~ small,
    .stTextInput div div div div small,
    .stTextInput [data-baseweb="base-input"] ~ div,
    .stTextInput div div div:last-child,
    .stTextInput div[data-baseweb="input"] small,
    .stTextInput div[data-baseweb="input"] div[data-testid="stFormHelperText"],
    .stTextInput [data-baseweb="input"] [data-testid="stFormHelperText"],
    .stTextInput .e1b2p2ww2 span,
    .stTextInput .e1b2p2ww2 div,
    .stTextInput [class*="helptext"],
    .stTextInput [class*="helpText"],
    .stTextInput div[data-testid="textInputHelperText"],
    .stTextInput [data-testid="textInputHelperText"] {
        display: none !important;
        opacity: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        position: absolute !important;
        overflow: hidden !important;
    }
    
    /* Style submit button to match input height exactly */
    .stButton button[kind="primary"] {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        border: none !important;
        height: 44px !important; /* Match input height */
        margin-top: 0 !important; /* Remove top margin for alignment */
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #1976D2 !important;
        box-shadow: 0 2px 6px rgba(30, 144, 255, 0.4) !important;
    }
    
    /* Custom styling for button */
    .stButton > button {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 20px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
        padding: 4px 15px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #2BA3EC !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Create better aligned columns for input and button
input_col, button_col = st.columns([5, 1])

with input_col:
    # Hiding "Press enter to apply" by using st.text_area instead of st.text_input
    st.text_area(
        "",  # No label
        key="user_input_area",  # Changed key to avoid conflicts
        placeholder="Ask your Twitter/X support question...",
        label_visibility="collapsed",  # Hide label
        max_chars=1000  # Set a max character limit
    )

with button_col:
    # Submit button with better vertical alignment
    submit_placeholder = st.empty()  # Create a placeholder at exact same level
    if submit_placeholder.button("Submit", on_click=submit_clicked, type="primary"):
        pass  # The on_click function handles the action

# Add custom CSS for sidebar styling with the requested color scheme
st.markdown("""
<style>
    /* Color scheme */
    :root {
        --dodger-blue: #1E90FF;
        --piction-blue: #2BA3EC;
        --yellow-banana: #EFEDCE;
        --middle-blue: #AAF1FF;
        --morning-blue: #EBFCFF;
    }
    
    /* Sidebar styling */
    .css-1cypcdb {
        background-color: var(--morning-blue) !important;
        border-right: 1px solid rgba(170, 241, 255, 0.2) !important;
    }
    
    /* Sidebar title */
    .css-17c4c22 h1 {
        color: var(--dodger-blue) !important;
    }
    
    /* Bullet styling */
    .question-bullet {
        color: var(--piction-blue);
        margin-right: 8px;
    }
    
    /* Question styling */
    .sidebar-question {
        padding: 8px 0;
        color: #333;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(170, 241, 255, 0.3);
    }
    
    /* Category header */
    .question-category {
        color: var(--dodger-blue);
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 8px;
        border-left: 3px solid var(--piction-blue);
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar header
st.sidebar.title("X Support Guide")
st.sidebar.markdown(f'<div style="background-color: #EBFCFF; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #1E90FF;"><p style="margin: 0; color: #333; font-size: 0.9rem;">Here are some common questions you can ask the assistant</p></div>', unsafe_allow_html=True)

# Group questions by category
st.sidebar.markdown('<div class="question-category">Account Access</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> How do I reset my Twitter password?</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> I lost access to my email, how can I recover my account?</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="question-category">Security</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> My account was hacked, what should I do?</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> What are the best ways to secure my X account?</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="question-category">Account Management</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> How do I change my Twitter handle?</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-question"><span class="question-bullet">•</span> How can I permanently delete my account?</div>', unsafe_allow_html=True)

# Add some space then a reminder
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(f'<div style="background-color: #EFEDCE; padding: 10px; border-radius: 5px; margin-top: 20px;"><p style="margin: 0; color: #333; font-size: 0.85rem;"><strong>Tip:</strong> You can type any Twitter/X support question in the main input field.</p></div>', unsafe_allow_html=True)

# Improved clear button with styling
if st.session_state.chat_history:
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.chatbot.clear_memory()
            st.experimental_rerun()

# Simpler footer that doesn't use fixed positioning
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="margin-top: 30px; text-align: center; padding-top: 15px; border-top: 1px solid #AAF1FF40;">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
        <span style="font-size: 20px; margin-right: 5px;">🐦</span>
        <span style="color: #2BA3EC; font-weight: 500;">X Support</span>
    </div>
    <div style="font-size: 0.7rem; color: #666; margin-bottom: 5px;">© 2025 X Support Assistant</div>
    <div style="font-size: 0.7rem; color: #666;">Powered by AI</div>
</div>
""", unsafe_allow_html=True)