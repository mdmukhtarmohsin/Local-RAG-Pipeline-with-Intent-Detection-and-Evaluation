# 📘 Local RAG Pipeline with Intent Detection and Evaluation

This project is a comprehensive customer support assistant that uses a local RAG (Retrieval-Augmented Generation) pipeline. It features intent detection, local and remote LLM support (Ollama and Gemini), and a full evaluation framework with A/B testing capabilities.

## ✨ Features

- **Intent Detection**: Classifies user queries into `technical_support`, `billing_account`, or `feature_request`.
- **Hybrid LLM Support**: Uses local models via Ollama (e.g., `gemma3:4b-it-qat`) with a fallback to the Gemini API.
- **Dynamic RAG Pipeline**: Retrieves relevant context from different document sources based on the detected intent.
- **Interactive Web UI**: A Gradio-based interface for easy interaction with the assistant.
- **Comprehensive Evaluation**: A full suite for evaluating the pipeline's performance on various metrics.
- **A/B Testing**: Built-in scripts to compare the performance of different LLMs (e.g., local vs. cloud).
- **Modular & Extendable**: Designed with a clear separation of concerns, making it easy to add new intents, documents, or models.

## 📁 Project Structure

```
rag_support_assistant/
├── app/
│   ├── main.py                # FastAPI server
│   ├── llm_wrapper.py         # Local LLM & Gemini logic
│   ├── intent_classifier.py   # Classifier
│   ├── retriever.py           # Chroma/FAISS vector retrieval
│   ├── prompt_templates.py    # Intent-based prompt templates
│   ├── router.py              # RAG routing logic
│   └── config.yaml
├── ui/
│   └── app_ui.py              # Gradio Web UI
├── eval/
│   ├── test_queries.json
│   ├── evaluator.py           # Evaluation script
│   ├── metrics.py             # Metrics functions
│   └── dashboard.py           # Gradio dashboard for results
├── data/
│   ├── billing_docs/
│   ├── tech_docs/
│   └── feature_requests/
├── embeddings/
│   └── vector_dbs/
├── README.md
├── requirements.txt
├── setup.sh
└── .env.example
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- `curl` for installing Ollama

### 2. Setup

Run the setup script to install all dependencies, download the necessary models, and prepare your environment.

```bash
bash setup.sh
```

The script will:

1.  Create a Python virtual environment in `venv/`.
2.  Install all required packages from `requirements.txt`.
3.  Install [Ollama](https://ollama.com/) if it's not already installed.
4.  Pull the `gemma3:4b-it-qat` model.
5.  Create a `.env` file from `.env.example`. You should edit this file to add your `GEMINI_API_KEY` if you want to use the Gemini model.
6.  Ask if you want to pre-build the knowledge base (vector database). This is recommended for the first run.

After setup, activate the virtual environment:

```bash
source venv/bin/activate
```

### 3. Building the Knowledge Base

If you skipped this step during setup, you can build the vector database for the RAG pipeline by running:

```bash
python3 -m app.retriever --build
```

This command will process the documents in the `data/` directory and create the necessary embeddings in `embeddings/vector_dbs/`.

## 🖥️ How to Run

The application consists of two main parts: the FastAPI backend and the Gradio UI. You need to run them in two separate terminals.

**Terminal 1: Run the Backend Server**

```bash
python3 app/main.py
```

The server will start on `http://localhost:8000`.

**Terminal 2: Run the Gradio Web UI**

```bash
python3 ui/app_ui.py
```

The UI will be available at `http://localhost:7860`.

Now you can open your browser and interact with the assistant!

## 🧪 Evaluation

The project includes a full suite for evaluating and A/B testing the RAG pipeline.

### 1. Run the Evaluation Script

To run the evaluation on the test queries defined in `eval/test_queries.json`, execute:

```bash
python3 eval/evaluator.py
```

This script will test both the `local` (Ollama) and `gemini` models, calculate all the performance metrics, and save the results to `eval/evaluation_results_detailed.csv` and `eval/evaluation_summary.csv`.

### 2. View the Evaluation Dashboard

To visualize the evaluation results, run the dashboard app:

```bash
python3 eval/dashboard.py
```

This will launch a Gradio dashboard on `http://localhost:7861`, where you can see charts and tables comparing the performance of the tested models.

## 🔧 Configuration & Extensibility

### Switching LLMs

You can switch the LLM provider in two ways:

1.  **In the UI**: Use the "Select LLM Provider" dropdown to choose between `auto`, `local`, or `gemini` for each query.
2.  **In the Config**: Change the `default_model` in `app/config.yaml` to set the default behavior.

### Extending Intents and Data

To add a new intent:

1.  Add a new directory with your documents under the `data/` folder.
2.  Update the `intent.classes` and `data_paths` sections in `app/config.yaml`.
3.  Add new prompt templates for the intent in `app/prompt_templates.py`.
4.  Add new test queries to `eval/test_queries.json`.

---

This project provides a solid foundation for building and evaluating sophisticated RAG-based assistants. Enjoy experimenting!
