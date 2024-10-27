#!/bin/bash

# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
nohup ollama serve &
echo "Ollama server started."
