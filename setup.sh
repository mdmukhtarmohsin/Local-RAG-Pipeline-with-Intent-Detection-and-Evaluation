#!/bin/bash

# Exit on any error
set -e

echo "--- Setting up Virtual Environment ---"
python3 -m venv venv
source venv/bin/activate

echo "--- Installing Python Dependencies ---"
pip install -r requirements.txt

echo "--- Setting up Ollama ---"
# Check if Ollama is installed
if ! command -v ollama &> /dev/null
then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

echo "--- Pulling Local LLM Model (gemma3:4b-it-qat) ---"
ollama pull gemma3:4b-it-qat

echo "--- Creating .env file from example ---"
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created. Please add your GEMINI_API_KEY to it."
else
    echo ".env file already exists."
fi


echo "--- Pre-building Knowledge Base (Optional but Recommended) ---"
# This step can take a while
read -p "Do you want to build the knowledge base now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Building knowledge base... This might take some time."
    python3 -m app.retriever --build
    echo "Knowledge base built successfully."
else
    echo "Skipping knowledge base build. You can run 'python3 -m app.retriever --build' later."
fi

echo "✅ Setup complete."

read -p "Do you want to start the application now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "--- Starting Application ---"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Start backend server in the background
    echo "Starting backend server... Logs are in logs/backend.log"
    nohup python3 app/main.py > logs/backend.log 2>&1 &
    BACKEND_PID=$!

    # Start UI in the background
    echo "Starting UI server... Logs are in logs/ui.log"
    nohup python3 ui/app_ui.py > logs/ui.log 2>&1 &
    UI_PID=$!

    echo "Application is running."
    echo "Backend (http://localhost:8000) is running with PID: $BACKEND_PID"
    echo "UI (http://localhost:7860) is running with PID: $UI_PID"
    echo "To stop the application, run: kill $BACKEND_PID $UI_PID"
else
    echo "To start the application later, activate the venv ('source venv/bin/activate') and run the servers in two separate terminals:"
    echo "1. Backend: python3 app/main.py"
    echo "2. UI: python3 ui/app_ui.py"
fi 