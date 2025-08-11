import logging
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document 



from .config import Config

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, vectorstore_path: Optional[Path] = None, embedding_model: str = None):
        self.config = Config()
        self.vectorstore_path = vectorstore_path or self.config.VECTORSTORE_PATH
        self.embedding_model = embedding_model or self.config.EMBEDDING_MODEL
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.retriever = self._create_retriever()
    
    def _initialize_embeddings(self):
        """Initialize embedding model (must match ingestion model)."""
        
        if self.embedding_model == "huggingface":
            model_name = self.config.EMBEDDING_MODEL_NAME
            logger.info(f"Using HuggingFace embedding model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def _load_vectorstore(self) -> FAISS:
        """Load existing vectorstore from disk."""
        if not self.vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at {self.vectorstore_path}. "
                "Please run document ingestion first."
            )
        
        try:
            vectorstore = FAISS.load_local(
                str(self.vectorstore_path), 
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Vectorstore loaded from: {self.vectorstore_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            raise
    
    def _create_retriever(self):
        """Create retriever from vectorstore."""
        return self.vectorstore.as_retriever(search_kwargs={"k": self.config.TOP_K_RESULTS})

    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        try:
            documents = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def similarity_search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        k = k or self.config.TOP_K_RESULTS
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Similarity search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_vectorstore_stats(self) -> dict:
        """Get statistics about the vectorstore."""
        try:
            # Get number of vectors
            index_to_docstore_id = self.vectorstore.index_to_docstore_id
            num_vectors = len(index_to_docstore_id) if index_to_docstore_id else 0
            
            # Get unique sources
            docstore = self.vectorstore.docstore
            sources = set()
            
            if hasattr(docstore, '_dict'):
                for doc in docstore._dict.values():
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
            
            return {
                "num_vectors": num_vectors,
                "num_unique_sources": len(sources),
                "sources": list(sources)
            }
        except Exception as e:
            logger.error(f"Error getting vectorstore stats: {e}")
            return {"error": str(e)}

def main():
    """CLI entry point for testing retriever."""
    Config.setup_logging()
    
    try:
        retriever = DocumentRetriever()
        
        # Test query
        test_query = "What are my skills?"
        documents = retriever.retrieve_documents(test_query)
        
        print(f"\n🔍 Query: {test_query}")
        print(f"📄 Retrieved {len(documents)} documents:")
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"\n{i}. Source: {source}")
            print(f"Content: {content_preview}")
        
        # Show stats
        stats = retriever.get_vectorstore_stats()
        print(f"\n📊 Vectorstore Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Retriever test failed: {e}")
        print(f"❌ Retriever test failed: {e}")

if __name__ == "__main__":
    main()