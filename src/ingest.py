import os
from pathlib import Path
from typing import List
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class PDFIngestor:
    def __init__(
        self,
        pdf_dir: str = "data/pdf",
        chroma_dir: str = "chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.chroma_dir = Path(chroma_dir)

        print("Text splitter initialization")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        print("Uploading embeddings model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Model uploaded")

    def load_pdfs(self) -> List:
        if not self.pdf_dir.exists():
            print(f" Directory {self.pdf_dir} not found")
            sys.exit(1)

        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No pdf founded in {self.pdf_dir}")

        print(f"{len(pdf_files)} pdf founded: ")
        for pdf in pdf_files:
            print(f" - {pdf.name}")

        all_documents = []
        print("Uploading PDF...")

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()

                all_documents.extend(documents)
                print(f" {pdf_path.name}: {len(documents)} pages")

            except Exception as e:
                print(f" Error with {pdf_path.name}: {e}")

        print(f"Total: {len(all_documents)} uploaded pages")
        return all_documents

    def chunk_documents(self, documents: List) -> List:
        print("Chunking ducuments...")

        chunks = self.text_splitter.split_documents(documents)

        print(f" Chunk created: {len(chunks)}")
        if documents:
            print(f" Chunk each page: {len(chunks)/len(documents)}")

        if chunks:
            print(" Example:")
            print(f" Digit: {len(chunks[0].page_content)}")
            print(f" Preview: {chunks[0].page_content[0:200]}...")

        return chunks

    def create_vectorstore(self, chunks: List):
        
        print("Creating VDB...")
        
        if Path(self.chroma_dir).exists():
            print(f" Database founded in {Path(self.chroma_dir)}")
            response = input(" Do you want to overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Operation failed")
                return
            import shutil
            shutil.rmtree(self.chroma_dir)
            print(" Previous DB eliminated")
            
        print("Embeddings creation")
        
        vectorstore = Chroma.from_documents(
            documents = chunks,
            embedding = self.embeddings,
            persist_directory = str(self.chroma_dir),
            collection_name = "Accademic_papers" 
        )                

        print("Vector Database created")
        
        return vectorstore

    def ingest(self):
        
        #Step 1: Upload PDF
        documents = self.load_pdfs()
        if not documents:
            print("No documents")
            return
        
        #Step 2: Chunking
        chunks = self.chunk_documents(documents)
        
        #Step 3: VectorDB creation
        vectorstore = self.create_vectorstore(chunks)
        
        print("Ingestion completed")
        print("Statistics: ")
        print(f"Processed PDF: {len(set(doc.metadata.get('source', '') for doc in documents))} ")
        print(f"Total pages: {len(documents)}")
        print(f"Chunk created: {len(chunks)}")
        print(f"Database: {self.chroma_dir}")

if __name__ == "__main__":
    ing = PDFIngestor(
        pdf_dir = "data/pdf",
        chroma_dir = "chroma_db",
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    
    ing.ingest()
