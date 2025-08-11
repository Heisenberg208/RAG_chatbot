# 🧠 Personal RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Python, FAISS vector search, and a modular architecture. This chatbot intelligently indexes your personal documents, retrieves the most relevant information for any query, and generates contextually informed responses using state-of-the-art language models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

### 🔍 **Intelligent Document Understanding**

- **Multi-format Support**: PDF, DOCX, TXT, MD, and image files (with OCR)
- **Smart Chunking**: Advanced text splitting with overlap for context preservation
- **Metadata Extraction**: File names, creation dates, and document structure

### 🚀 **Advanced RAG Pipeline**

- **Semantic Search**: FAISS-powered vector similarity search
- **Context-Aware Responses**: Combines retrieved documents with conversation history
- **Source Citation**: Tracks and displays document sources for transparency

### 🤖 **Flexible LLM Support**

- **Local Models**: Llama 2 and other ollama models
- **Cloud APIs**: Google Gemini
- **Privacy-First**: Run completely offline with local models

### ⚡ **Performance & Scalability**

- **Efficient Indexing**: Incremental document processing
- **Fast Retrieval**: Sub-second search across thousands of documents
- **Memory Management**: Optimized for both CPU and GPU environments

---

## 📂 Project Architecture

```bash
RAG_Chatbot/
├── 📁 data/                     # Your documents (PDFs, DOCX, images, etc.)
│   ├── documents/
│   ├── images/
│   └── processed/
├── 📁 src/                      # Core application code
│   ├── __init__.py
│   ├── 💬 chat.py              # Main chat interface & conversation logic
│   ├── ⚙️ config.py            # Configuration management & environment setup
│   ├── 📥 ingest.py            # Document processing & embedding generation
│   ├── 🖥️ interface.py         # Streamlit Web UI
│   └── 🔍 retriever.py         # Vector search & document retrieval
├── 📁 vectorstore/             # FAISS indexes & document metadata
│   ├── index.faiss             # Vector embeddings index
│   ├── index.pkl               # Document metadata
│   └── config.json             # Index configuration
├── 📁 logs/                    # Application logs
├── .env                        # Environment variables (secrets)
├── .gitignore                  # Git exclusions
├── poetry.lock                 # Dependency lock file
├── pyproject.toml             # Project configuration
└── README.md
```

---

## 🚀 Quick Start Guide

### 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **8GB+ RAM** (16GB+ for local LLMs)
- **GPU** (optional, for faster inference)

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Heisenberg208/RAG_Chatbot.git
cd RAG_Chatbot

# Install with Poetry (recommended)
poetry install

# Or install with pip
pip install -r requirements.txt
```

### 2️⃣ Configuration Setup

To be able run the Chatbot you need to create `.env` file and `data/`(which is not pushed in repo) folder and add relevant documents based on which you want your RAG chatbot to assist on.

Create your `.env` file:

```ini
# LLM Provider Configuration
# Depending on requirement(flexible for both local ollama (free) and gemini models)
LLM_PROVIDER=o llama #change to `gemini` for Gemini integration  
LLM_MODEL=lama3.2:3b


# API Keys (if using cloud models)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash

# Embedding Configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2

WEB_URLS = https.example1.com,https.example2.com
# Vector Store Settings
VECTOR_STORE_PATH=./vectorstore


# Logging
LOG_LEVEL=INFO
```

### 3️⃣ Add Your Documents

```bash
# Create data directory structure
mkdir -p data/{documents,images,processed}

# Add your files to data/documents/
cp /path/to/your/files/* data/documents/

# Supported formats: PDF, DOCX, TXT, MD, PNG, JPG, etc.
```

### 4️⃣ Ingest Your Documents

```bash
# Process and index all documents
poetry run python -m src.ingest # Note running python src/ingest.py would throw a module not found error

```

### 🎯 Custom Embedding Models

```python
# In  .env
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Better quality
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # Faster, lighter
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"                  # Multilingual
```

### 🧠 LLM Model Options

TO run ollama model first you need to pull the model(provided ollama is already installed in your system) with `ollama pull lama3.2:3b`

```ini
# Local Models (100% local)
LLM_PROVIDER=ollama
LLM_MODEL=lama3.2:3b # Note to run ollama 
# OR
# Cloud Models (Higher quality, requires API key)
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-pro


```

### 5️⃣ Start Chatting

```bash
# Launch interactive CLI
poetry run python -m src.chat

# Or start the web interface
poetry run streamlit src/interface.py 

```

---

## 💡 Usage Examples

### 📱 Interactive CLI

Note: Change the PROMPT in `chat.py`according to the documents in the data folder.

```bash
🤖 Personal RAG Chat initialized with gemini!
📚 Indexed 1,247 documents from your collection

❓ Ask a question: What are the key findings from my research papers on AI?

🔍 Searching documents...
🤖 Answer: Based on your research papers, the key findings on AI include:

1. **Neural Architecture Search** (Paper: "AutoML_Survey_2023.pdf"):
   - Automated model design reduces human bias by 34%
   - Performance improvements of 15-20% over manual designs

2. **Transformer Efficiency** (Paper: "Efficient_Transformers.pdf"):
   - Linear attention mechanisms reduce computational complexity
   - Memory usage decreased by 40% while maintaining accuracy

📚 Sources (3 documents):
1. AutoML_Survey_2023.pdf - "Our experiments show that neural architecture search..."
2. Efficient_Transformers.pdf - "We propose a novel linear attention mechanism..."
3. AI_Ethics_Framework.pdf - "The implications of automated AI development..."


```

### 🌐 Web Interface

```bash
# Start web server
poetry run streamlit run src/interface.py

```

## 🔒 Security & Privacy

### 🛡️ Data Protection

- **Local Processing**: All documents stay on your machine
- **No Data Transmission**: Optional cloud LLMs only receive query context
- **Encrypted Storage**: Vector indices can be encrypted at rest

### 🌟 Showcase

Share your RAG chatbot implementations:

- **Academic Research**: Document analysis and literature review
- **Corporate Knowledge**: Internal documentation and training materials  
- **Personal Productivity**: Note organization and information retrieval
- **Legal Practice**: Case law research and document review

---
