import subprocess
import socket
import time
import ollama

OLLAMA_PORT = 11434
MODEL = "gemma:2b"

def is_port_in_use(port):
    """Check if a given port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def ensure_ollama_running():
    """Start Ollama server if not already running."""
    if is_port_in_use(OLLAMA_PORT):
        print(f"Ollama server already running on port {OLLAMA_PORT}")
        return None
    else:
        print("Starting Ollama server...")
        process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for server to boot
        return process

if __name__ == "__main__":
    server_process = ensure_ollama_running()

    print(f"{MODEL} running. What is your question? Type 'exit' to quit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        response = ollama.generate(model=MODEL, prompt=user_input)
        print(response['response'].replace('*', ''))

    # Stop only if we started it in this script
    if server_process:
        server_process.terminate()