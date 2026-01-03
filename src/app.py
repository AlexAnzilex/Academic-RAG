"""
app.py - Streamlit Web UI for RAG system

WHAT IT DOES:
User-friendly GUI to interact with RAG
- Input question
- Display answer
- Show sources and retrieved chunks
- Conversation history

INSTALLATION:
pip install streamlit

USAGE:
streamlit run src/app.py

BROWSER:
Automatically opens http://localhost:8501
"""

import streamlit as st
from pathlib import Path
import time
from typing import Dict, List
import sys

# Ensure project root is on sys.path when run via `streamlit run src/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag import AcademicRAG


# ============= PAGE CONFIGURATION =============

st.set_page_config(
    page_title="Academic RAG",
    page_icon="📚",
    layout="wide", 
    initial_sidebar_state="expanded"
)


# ============= HELPER FUNCTIONS =============

@st.cache_resource
def load_rag_system():
    """
    Load RAG system (cached to avoid reload every time)
    
    @st.cache_resource: Cache that persists between reloads
    Useful for heavy objects like ML models
    """
    try:
        with st.spinner("Loading RAG system..."):
            rag = AcademicRAG(
                chroma_dir="chroma_db",
                model="llama3.1:8b",
                top_k=5
            )
            rag.load_vectorstore()
        return rag
    except Exception as e:
        st.error(f"Error during loading: {e}")
        st.info(" Did you run `python src/ingest.py` to create database?")
        st.stop()


def format_sources(sources: List[str]) -> str:
    """
    Format sources for display
    """
    return ", ".join([Path(src).name for src in sources])


def display_context_card(ctx: Dict, index: int):
    """
    Show retrieved chunk in a card
    """
    source = Path(ctx['metadata'].get('source', 'unknown')).name
    page = ctx['metadata'].get('page', '?')
    score = ctx['similarity_score']
    content = ctx['content']
    
    with st.expander(f"📄 Chunk {index} - {source} (page {page}) - Score: {score:.3f}"):
        st.markdown(f"**Source:** {source}")
        st.markdown(f"**Page:** {page}")
        st.markdown(f"**Similarity Score:** {score:.3f}")
        st.markdown("---")
        st.text(content[:500] + ("..." if len(content) > 500 else ""))


# ============= STATE INITIALIZATION =============

# Session state to maintain history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = load_rag_system()


# ============= SIDEBAR =============

with st.sidebar:
    st.title("⚙️ Settings")
    
    # System info
    st.markdown("### System")
    
    try:
        chunks_count = st.session_state.rag_system.vectorstore._collection.count()
        st.metric("Chunks in DB", chunks_count)
    except:
        st.warning("⚠️ Database not available")
    
    st.markdown(f"**Model:** {st.session_state.rag_system.model}")
    
    # Query parameters
    st.markdown("### Parameters")
    
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most relevant chunks to retrieve"
    )
    
    show_contexts = st.checkbox(
        "Show retrieved chunks",
        value=False,
        help="Display chunks used to generate answer"
    )
    
    # Button to clear chat
    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Info
    st.markdown("---")
    st.markdown("###Info")
    st.markdown("""
    **Academic RAG** lets you ask questions about your academic PDFs.
    
    The system:
    1. Searches most relevant chunks
    2. Passes them to Ollama
    3. Generates accurate answer
    
    **Commands:**
    - Write question and press Enter
    - Use sidebar to change parameters
    """)


# ============= MAIN CONTENT =============

# Header
st.title("Academic RAG System")
st.markdown("Ask questions about your academic PDFs using local AI (Ollama)")

# Divider
st.markdown("---")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If it's assistant response, show sources
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                st.markdown(message["sources"])
            
            # Show contexts if requested
            if show_contexts and "contexts" in message:
                with st.expander("🔍 Retrieved chunks"):
                    for i, ctx in enumerate(message["contexts"], 1):
                        display_context_card(ctx, i)

# User input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
 
 
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                
                start_time = time.time()
                result = st.session_state.rag_system.query(prompt, k=top_k)
                elapsed_time = time.time() - start_time
                
                st.markdown(result['answer'])
                
                # Info box
                sources_text = format_sources(result['sources'])
                st.info(f" **Sources:** {sources_text} |  **Time:** {elapsed_time:.2f}s")
                
                # Save in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": sources_text,
                    "contexts": result['contexts'] if show_contexts else None
                })
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An error occurred: {e}"
                })


# ============= FOOTER =============

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Tip:** Ask specific questions for more accurate answers")

with col2:
    if st.session_state.messages:
        st.markdown(f" **Messages:** {len(st.session_state.messages)}")

with col3:
    st.markdown(" **Powered by:** Ollama + LangChain")


# ============= MAIN (optional) =============

if __name__ == "__main__":
    # This runs only if you launch file directly
    # With streamlit run it's not needed
    pass
