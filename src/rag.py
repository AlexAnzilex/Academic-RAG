import os
from typing import List, Dict, Optional
from pathlib import Path
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests


class AcademicRAG:
    
    
    def __init__(
        self, 
        chroma_dir: str = "chroma_db",
        model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
        top_k: int = 5
    ):
        
        self.chroma_dir = chroma_dir
        self.model = model
        self.ollama_url = ollama_url
        self.top_k = top_k
        
        
        print("Uploading embeddings..")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(" Verifing Ollama connection ")
        self._check_ollama()

        self.vectorstore = None

        print("RAG initialization complited")

    def _check_ollama(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama non risponde")

            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if self.model not in model_names:
                print(f"\n⚠️ Model '{self.model}' not found")
                print(f"   Available models: {', '.join(model_names)}")

                if model_names:
                    print(f"   Do you want to use '{model_names[0]}' ? (y/n)")
                    choice = input("    > ").strip().lower()
                    if choice == 'y':
                        self.model = model_names[0]
                        print(f"    Using {self.model}")
                    else:
                        raise ValueError(f"Modello {self.model} non disponibile")
                else:
                    raise ValueError("Nessun modello Ollama installato!")

        except requests.exceptions.ConnectionError:
            print("Ollama is not working")
            raise

    def load_vectorstore(self):
        if not Path(self.chroma_dir).exists():
            raise FileNotFoundError(f"None file in the path {Path(self.chroma_dir)}")

        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings,
            collection_name="Accademic_papers",
        )

        collection = self.vectorstore._collection
        count = collection.count()

        print(f"Vectore DB uploaded: {count} chunks available")

        return self.vectorstore

    def retrieval(self, query: str, k: Optional[int] = None) -> List[Dict]:
        if not self.vectorstore:
            raise ValueError("Vectorstore not upladed")

        if k is None:
            k = self.top_k

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        contexts = []

        for i, (doc, score) in enumerate(results, 1):
            contexts.append(
                {
                    'rank': i,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': 1 - score,
                }
            )

            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            print(f"  {i}. {Path(source).name} (pag. {page}) - score: {1-score:.3f}")

        return contexts

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = "\n\n---\n\n".join(
            [
                f"[Doc {i}] (Fonte: {ctx['metadata'].get('source', 'unknown')}, "
                f"Pag: {ctx['metadata'].get('page', '?')})\n{ctx['content']}"
                for i, ctx in enumerate(contexts, 1)
            ]
        )

        prompt = f"""You are an expert assistant. Answer the question using ONLY the information provided in the CONTEXT.

            RULES:
            - Use only information from the context
            - If the answer cannot be found, explicitly say so
            - Be precise and concise
            - Cite sources when relevant

            CONTEXT:
            {context_text}

            QUESTION: {query}

            ANSWER:"""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 1000,
                    },
                },
                timeout=120,
            )

            if response.status_code != 200:
                return f"Ollama Error: {response.status_code}"

            result = response.json()
            answer = result.get('response', '').strip()

            if 'total_duration' in result:
                duration_sec = result['total_duration'] / 1e9
                print(f" Time : {duration_sec:.1f}s")

            return answer

        except requests.exceptions.Timeout:
            return "Ollama is to slow"
        except requests.exceptions.ConnectionError:
            return "Connection Error"
        except Exception as e:
            return f"Error: {e}"

    def query(self, question: str, k: Optional[int] = None) -> Dict:
        print(f"Question {question}")

        contexts = self.retrieval(question, k=k)

        if not contexts:
            return {
                'answer': "No relevant information",
                'contexts': [],
                'sources': [],
            }

        prompt = self.build_prompt(question, contexts)

        answer = self.generate_answer(prompt)

        sources = list(
            set(
                [
                    ctx['metadata'].get('source', 'unknown')
                    for ctx in contexts
                ]
            )
        )

        return {
            'answer': answer,
            'contexts': contexts,
            'sources': sources,
        }

    def chat(self):
        print("\n" + "=" * 60)
        print("INTERACTIVE RAG CHAT (powered by Ollama)")
        print("=" * 60)
        print("Available commands:")
        print("  - 'exit' or 'quit': Exit")
        print("  - 'sources': Show sources of the last answer")
        print("  - 'model': Change Ollama model")
        print("  - Otherwise: Ask a question!")
        print("=" * 60 + "\n")

        last_result = None

        while True:
            question = input("Question: ").strip()

            if question.lower() in ['exit', 'quit']:
                print("Bye")
                break

            if question.lower() == 'source':
                if last_result:
                    print("Used sources")
                    for src in last_result['sources']:
                        print(f"  - {Path(src).name}")
                else:
                    print("No query executed")
                continue

            if question.lower() == 'model':
                print(f"\nCurrent model: {self.model}")
                new_model = input("New model: ").strip()
                if new_model:
                    self.model = new_model
                    print(f"Model changed: {self.model}")
                continue

            if not question:
                continue

            try:
                result = self.query(question)
                last_result = result

                print(f"Ollama: {result['answer']}")
                print(f"Sources: {', '.join([Path(s).name for s in result['sources']])}")

            except Exception as e:
                print(f" Errore: {e}")
                    
    
if __name__ == "__main__":
        
    rag = AcademicRAG(
        chroma_dir="chroma_db",
        model="llama3.2:3b", 
        top_k=5
        )
    
    rag.load_vectorstore()
    rag.chat()
