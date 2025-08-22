import streamlit as st
import ollama
import subprocess
import socket
import time
from datetime import datetime

# Configuration
OLLAMA_PORT = 11434
MODEL = "gemma:2b"

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

def chat_with_gemma(prompt):
    """Send a prompt to Gemma and get response."""
    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
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
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for model info and controls
    with st.sidebar:
        st.header("Model Information")
        st.info(f"**Model:** {MODEL}")
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
            
            # Get response from Gemma
            response = chat_with_gemma(prompt)
            
            # Update placeholder with actual response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Powered by Ollama • {MODEL} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Cleanup when app is closed
    if server_process:
        st.session_state.server_process = server_process

if __name__ == "__main__":
    main()
