# RAG on Academic PDFs

Simple pipeline to ingest PDFs, index them in Chroma with Hugging Face embeddings, and query via an Ollama model (e.g., `llama3.2:3b`).

## Prerequisites
- Python 3.10+
- Ollama running with a model available (default `llama3.2:3b`)
- Python deps: `pip install -r requirements.txt`

## Quick setup
```bash
python -m venv venv-rag
source venv-rag/bin/activate
pip install -r requirements.txt
```

## PDF ingestion
1) Drop PDFs into `data/pdf/`.  
2) Run ingest:  
```bash
python -m src.ingest
```
This creates/updates the Chroma database in `chroma_db/` with indexed chunks.

## RAG query & chat
Launch the CLI chat:
```bash
python -m src.rag
```
- Checks Ollama connectivity and model.  
- Loads the vector store from `chroma_db/`.  
- Interactive loop commands: `model` to switch model, `sources` to show last answer’s sources, `exit`/`quit` to leave.

## Main configuration
`src/ingest.py`:
- `pdf_dir`, `chroma_dir`: input/output paths.
- `chunk_size`, `chunk_overlap`: splitter params.

`src/rag.py`:
- `chroma_dir`: DB directory.
- `model`, `ollama_url`: Ollama model and endpoint.
- `top_k`: number of chunks to retrieve.

## Notes
- `venv-rag/`, `chroma_db/`, input PDFs, and `.env` are git-ignored.
- `HuggingFaceEmbeddings` is deprecated in LangChain 0.2.2; consider `langchain-huggingface` if you upgrade.
