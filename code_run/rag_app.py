import os
import json
import glob
from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
import numpy as np

# LangChain imports (fixed versions)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    print("✅ All LangChain imports successful!")
except ImportError as e:
    st.error(f"❌ Missing package: {e}")
    st.info("Run: `pip install langchain-huggingface langchain-community faiss-cpu`")
    st.stop()

class HindiEnglishRAG:
    def __init__(self):
        self.data_folder = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents_count = 0
    
    def set_data_path(self, data_path: str):
        """Set data path and build index"""
        self.data_folder = data_path
        self._build_index()
    
    def _load_all_json(self, folder_path: str) -> List[Document]:
        """Load all JSON files recursively"""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            return documents
        
        json_files = list(folder.rglob("*.json"))
        st.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle both list and dict
                    if isinstance(data, list):
                        for item in data:
                            doc = self._json_to_document(item, str(json_file))
                            documents.append(doc)
                    else:
                        doc = self._json_to_document(data, str(json_file))
                        documents.append(doc)
                        
            except Exception as e:
                st.warning(f"Error reading {json_file}: {e}")
        
        return documents
    
    def _json_to_document(self, item: Dict, source_file: str) -> Document:
        """Convert JSON item to Document"""
        content_parts = []
        
        # Extract meaningful content
        keys_to_extract = ['plant_name', 'name', 'hindi_name', 'scientific_name', 
                          'uses', 'benefits', 'diseases', 'treatment', 'description']
        
        for key in keys_to_extract:
            if key in item:
                value = item[key]
                if isinstance(value, (str, list)):
                    content_parts.append(f"{key.replace('_', ' ').title()}: {str(value)}")
        
        # Add all other fields
        for key, value in item.items():
            if key not in keys_to_extract and isinstance(value, str):
                content_parts.append(f"{key}: {value}")
        
        content = "\n".join(content_parts)
        
        metadata = {
            'source': Path(source_file).name,
            'plant_name': item.get('plant_name') or item.get('name', ''),
            'hindi_name': item.get('hindi_name', ''),
            'scientific_name': item.get('scientific_name', '')
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _build_index(self):
        """Build FAISS index from JSON files"""
        with st.spinner("Loading JSON files..."):
            documents = self._load_all_json(self.data_folder)
            self.documents_count = len(documents)
        
        if not documents:
            st.error("No JSON documents found!")
            return
        
        with st.spinner("Creating embeddings..."):
            # Use multilingual embeddings for Hindi+English
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            splits = splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        st.success(f"✅ Index built! {self.documents_count} documents loaded")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search the vector store"""
        if not self.vectorstore:
            return []
        
        # Similarity search with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        results = []
        
        for doc, score in docs_with_scores:
            similarity = max(0, 1 - (score / 0.5))  # Normalize score
            results.append({
                'content': doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content,
                'metadata': doc.metadata,
                'similarity': round(similarity, 3)
            })
        
        return results

def main():
    st.set_page_config(page_title="🌿 Plant RAG", page_icon="🌱", layout="wide")
    
    st.title("🌿 Plant Metadata RAG System")
    st.markdown("**Search your `data\\plant_metadata\\metadata` JSON files**")
    
    # Path Configuration
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        data_path = st.text_input(
            "📁 JSON Folder Path",
            value="data\\plant_metadata\\metadata",
            help="Supports Windows paths with \\ and recursive search"
        )
    
    with col2:
        if st.button("🚀 Load & Index", type="primary", use_container_width=True):
            rag = HindiEnglishRAG()
            rag.set_data_path(data_path)
            st.session_state.rag = rag
            st.session_state.data_path = data_path
            st.rerun()
    
    # Show loaded data info
    if 'rag' in st.session_state:
        rag = st.session_state.rag
        col1, col2 = st.columns(2)
        col1.metric("📄 Documents", rag.documents_count)
        col2.metric("🧠 Status", "✅ Indexed")
        st.markdown("---")
        
        # Search Tabs
        tab1, tab2 = st.tabs(["🔍 Plant Search", "💊 Remedy Search"])
        
        with tab1:
            query = st.text_input("Search plant/disease:", placeholder="नीम, diabetes, Tulsi...")
            if st.button("Search", type="primary") and query:
                results = rag.search(query, k=5)
                for i, result in enumerate(results, 1):
                    with st.expander(f"🌱 {i}. {result['metadata'].get('plant_name', 'N/A')} - {result['similarity']:.1%}"):
                        st.json(result['metadata'])
                        st.markdown(result['content'])
        
        with tab2:
            issue = st.text_input("Health issue:", placeholder="cold, सर्दी, diabetes...")
            if st.button("Find Top 5 Plants", type="primary") and issue:
                results = rag.search(f"plants medicine treatment for {issue}", k=5)
                for i, result in enumerate(results, 1):
                    with st.expander(f"💊 #{i} - {result['similarity']:.1%}"):
                        st.markdown(result['content'])

if __name__ == "__main__":
    main()