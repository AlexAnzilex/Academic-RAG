# 📚 Academic RAG System

A Retrieval-Augmented Generation (RAG) system for querying academic PDFs using local AI models. Built with LangChain, ChromaDB, and Ollama

## ✨ Features

- 🔍 **Semantic Search** - Find relevant information across multiple PDFs
- 🤖 **Local AI** - Uses Ollama (no API costs, complete privacy)
- ⚡ **Fast Retrieval** - ChromaDB vector database for efficient similarity search
- 🎨 **Multiple Interfaces** - CLI, REST API, and Web UI
- 📊 **Source Attribution** - Shows which PDF and page the answer comes from
- 🔄 **Persistent Storage** - Vector database saved on disk

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF Files │────▶│  Ingest.py   │────▶│  ChromaDB   │
└─────────────┘     │ (Embedding)  │     │ (Vectors)   │
                    └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐            │
                    │   Query      │◀───────────┘
                    │   + Ollama   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐      ┌──────▼─────┐    ┌──────▼─────┐
   │   CLI   │      │  REST API  │    │   Web UI   │
   │ (rag.py)│      │  (api.py)  │    │  (app.py)  │
   └─────────┘      └────────────┘    └────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed and running

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/Academic-RAG.git
cd Academic-RAG.git
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and start Ollama**
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Download model
ollama pull llama3.1:8b

# Start Ollama (usually automatic)
ollama serve
```

5. **Add your PDFs**
```bash
# Place your PDF files in data/pdf/
mkdir -p data/pdf
cp your_papers.pdf data/pdf/
```

6. **Create vector database**
```bash
python src/ingest.py
```

### Usage

#### Option 1: CLI (Interactive Chat)
```bash
python src/rag.py
```

#### Option 2: REST API
```bash
# Start server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does attention mechanism work?"}'

# Or visit interactive docs
open http://localhost:8000/docs
```

#### Option 3: Web UI (Streamlit)
```bash
streamlit run src/app.py
```

## 📁 Project Structure

```
academic-rag/
├── data/
│   └── pdf/                    # Your PDF files go here
├── src/
│   ├── __init__.py
│   ├── ingest.py              # PDF ingestion and embedding
│   ├── rag.py                 # RAG query engine (CLI)
│   ├── api.py                 # FastAPI REST API
│   └── app.py                 # Streamlit web interface
├── chroma_db/                 # Vector database (created by ingest.py)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Changing Models

Edit model in `src/rag.py`, `src/api.py`, or `src/app.py`:

```python
rag = AcademicRAG(
    model="llama3.1:8b",  # Options: llama3.1:8b, mistral:7b, phi3:mini
    top_k=5               # Number of chunks to retrieve
)
```

### Chunk Size

Adjust chunking parameters in `src/ingest.py`:

```python
ingestor = PDFIngestor(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

## 📊 API Endpoints

### `POST /query`
Query the RAG system

**Request:**
```json
{
  "question": "How does attention mechanism work?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The attention mechanism is...",
  "sources": ["attention.pdf", "transformer.pdf"],
  "contexts_count": 5,
  "processing_time": 2.3
}
```

### `GET /health`
Check system health

**Response:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "vectordb_loaded": true,
  "chunks_count": 152
}
```

### `GET /stats`
Database statistics

**Response:**
```json
{
  "total_chunks": 152,
  "total_pdfs": 4,
  "model": "llama3.1:8b",
  "collection_name": "academic_papers"
}
```

## 🎯 Use Cases

- 📖 **Research** - Query multiple papers simultaneously
- 🎓 **Study Aid** - Quick answers from textbooks and lecture notes
- 📝 **Literature Review** - Find relevant information across papers
- 🔬 **Lab Notes** - Search through experimental documentation

## 🛠️ Tech Stack

- **LangChain** - RAG orchestration framework
- **ChromaDB** - Vector database for embeddings
- **Ollama** - Local LLM inference
- **HuggingFace** - Sentence transformers for embeddings
- **FastAPI** - REST API framework
- **Streamlit** - Web UI framework
- **PyPDF** - PDF text extraction

## 🚧 Roadmap

- [ ] Support for multiple languages
- [ ] Add document filters (by date, author, etc.)
- [ ] Implement conversation memory
- [ ] Add citation extraction
- [ ] Support for images and tables
- [ ] Export chat history
- [ ] Docker deployment

## 📝 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Alex Anzile - [@alexanzilex](www.linkedin.com/in/alexanzile)

Project Link: [https://github.com/yourusername/Academic-RAG](https://github.com/yourusername/Academic-RAG)

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [Ollama](https://ollama.com/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector database


---

⭐ If you find this project useful, please consider giving it a star!
