import streamlit as st
import ollama
import subprocess
import socket
import time
from datetime import datetime

# Configuration
OLLAMA_PORT = 11434
AVAILABLE_MODELS = ["gemma:2b", "deepseek-coder:6.7b", "llama2:7b", "mistral:7b"]

def is_port_in_use(port):
    """Check if a given port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def ensure_ollama_running():
    """Start Ollama server if not already running."""
    if is_port_in_use(OLLAMA_PORT):
        st.success(f"✅ Ollama server already running on port {OLLAMA_PORT}")
        return None
    else:
        st.info("🚀 Starting Ollama server...")
        try:
            process = subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)  # Wait for server to boot
            st.success("✅ Ollama server started successfully!")
            return process
        except FileNotFoundError:
            st.error("❌ Ollama not found. Please install Ollama first: https://ollama.ai/")
            return None

def chat_with_llm(model, prompt):
    """Send a prompt to the selected LLM and get response."""
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response'].replace('*', '')
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Gemma Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Gemma Chat")
    st.markdown("Chat with Google's Gemma 2B language model powered by Ollama")
    
    # Initialize Ollama
    server_process = ensure_ollama_running()
    
    # Initialize chat history and model selection
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemma:2b"
    
    # Sidebar for model selection and controls
    with st.sidebar:
        st.header("Model Selection")
        
        # Model dropdown
        selected_model = st.selectbox(
            "Choose your LLM:",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model)
        )
        
        # Update selected model
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.messages = []  # Clear chat when switching models
            st.rerun()
        
        st.header("Model Information")
        st.info(f"**Model:** {st.session_state.selected_model}")
        st.info(f"**Port:** {OLLAMA_PORT}")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Type your message below")
        st.markdown("2. Press Enter or click Send")
        st.markdown("3. Gemma will respond with AI-generated text")
        st.markdown("4. Use 'Clear Chat History' to start fresh")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask Gemma anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")
            
            # Get response from selected LLM
            response = chat_with_llm(st.session_state.selected_model, prompt)
            
            # Update placeholder with actual response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Powered by Ollama • {st.session_state.selected_model} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Cleanup when app is closed
    if server_process:
        st.session_state.server_process = server_process

if __name__ == "__main__":
    main()
