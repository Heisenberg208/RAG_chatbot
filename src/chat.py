import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
# For local transformers models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not available. Local models won't work.")

from .retreiver import DocumentRetriever
from .config import Config

PROMPT="""You are an AI assistant helping answer questions about personal documents and information. 
    Use the provided context to answer the user's question accurately and comprehensively.

    Context from personal documents:
    {context}

    Chat History:
    {chat_history}

    User Question: {question}

    Instructions:
    1. Answer based primarily on the provided context from personal documents
    2. If the context doesn't contain enough information, clearly state what information is missing
    3. Be specific and cite relevant details from the documents when possible
    4. If referencing previous conversation, use the chat history appropriately
    5. Maintain a helpful and professional tone

    Answer:"""


class PersonalRAGChat:
    def __init__(self, retriever: Optional[DocumentRetriever] = None):
            """Initialize the Personal RAG Chat system."""
            logger.info("🚀 Initializing Personal RAG Chat system...")
            
            self.config = Config()
            logger.info(f"📋 Configuration loaded - LLM Provider: {self.config.LLM_PROVIDER}")
            
            self.retriever = retriever or DocumentRetriever()
            logger.info("📚 Document retriever initialized")
            
            # Initialize LLM (this will log which model is being used)
            self.llm = self._initialize_llm()
            
            self.memory = ConversationBufferWindowMemory(
                k=5,  # Keep last 5 exchanges
                memory_key="chat_history",
                return_messages=True
            )
            logger.info("💭 Conversation memory initialized (keeping last 5 exchanges)")
            
            self.qa_chain = self._create_qa_chain()
            logger.info("🔗 QA chain created successfully")
            
            logger.info("✅ Personal RAG Chat system fully initialized and ready!")
    def _initialize_llm(self):
        """Initialize the language model based on provider."""
        provider = self.config.LLM_PROVIDER.lower()
        
        if  provider == "ollama":
            # Use Ollama to run Mistral 7B locally 
            model_name = "llama3.2:3b"
            logger.info(f"🤖 Initializing Ollama with model: {model_name}")
            logger.info(f"🏠 Using local {model_name} model via Ollama (free)")
            
            llm = Ollama(
                model=model_name,
                temperature=0.1,
                num_ctx=4096,  # Context window
                num_predict=1000,  # Max tokens to generate
            )
            
            logger.info(f"✅ Ollama/{model_name} model initialized successfully")
            return llm
            
        elif provider == "gemini":
            if not self.config.GEMINI_API_KEY:
                logger.error("❌ Google API key not found in environment")
                raise ValueError("Google API key not found in environment.")
            
            model_name = self.config.GEMINI_MODEL or "gemini-2.5-flash"
            logger.info(f"🤖 Initializing Google Gemini with model: {model_name}")
            logger.info("☁️ Using Google Gemini API (requires API key)")
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.config.GEMINI_API_KEY,
                temperature=0.1,
                max_tokens=1000
            )
            
            logger.info("✅ Google Gemini model initialized successfully")
            return llm
            
        else:
            logger.error(f"❌ Unsupported LLM provider: {provider}")
            logger.error("📝 Supported providers: 'mistral', 'ollama', 'gemini'")
            raise ValueError(f"Unsupported LLM provider: {provider}")


    def _create_qa_chain(self):
        """Create the question-answering chain using newer LangChain patterns."""
        prompt_template = PROMPT
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # For newer LangChain versions, you might want to use this pattern:
        from langchain.chains import LLMChain
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        return chain

    def ask_question(self, question: str, include_sources: bool = True) -> Dict:
        """Alternative implementation using direct LLM call."""
        try:
            logger.info(f"Processing question: {question}")
            
            # Retrieve relevant documents
            documents = self.retriever.retrieve_documents(question)
            context = self._format_documents(documents)
            
            # Get chat history from memory
            chat_history = ""
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                messages = self.memory.chat_memory.messages
                chat_history = "\n".join([f"Human: {msg.content}" if hasattr(msg, 'content') else str(msg) 
                                        for msg in messages[-10:]])  # Last 10 messages
            
            # Prepare the prompt
            prompt_text = f"""You are an AI assistant helping answer questions about personal documents and information. 
    Use the provided context to answer the user's question accurately and comprehensively.

    Context from personal documents:
    {context}

    Chat History:
    {chat_history}

    User Question: {question}

    Instructions:
    1. Answer based primarily on the provided context from personal documents
    2. If the context doesn't contain enough information, clearly state what information is missing
    3. Be specific and cite relevant details from the documents when possible
    4. If referencing previous conversation, use the chat history appropriately
    5. Maintain a helpful and professional tone

    Answer:"""
            
            # Generate response
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt_text)
            else:
                response = self.llm(prompt_text)
            
            # Extract response text
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            # Save to memory
            self.memory.save_context({"input": question}, {"output": answer_text})
            
            # Prepare result
            result = {
                "question": question,
                "answer": answer_text.strip(),
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(documents)
            }
            
            if include_sources:
                sources = []
                for doc in documents:
                    source_info = {
                        "file_name": doc.metadata.get('file_name', 'Unknown'),
                        "source_path": doc.metadata.get('source', 'Unknown'),
                        "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    }
                    sources.append(source_info)
                result["sources"] = sources
            
            logger.info("Question processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        if not documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown source')
            file_name = doc.metadata.get('file_name', 'Unknown file')
            content = doc.page_content.strip()
            
            formatted_doc = f"Document {i} (Source: {file_name}):\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n---\n\n".join(formatted_docs)
    
    def _rephrase_query(self, question: str) -> str:
        """Optionally rephrase the user query for better retrieval."""
        # Simple query expansion/rephrasing
        # This can be enhanced with a separate LLM call for query refinement
        return question
    

    
    def clear_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history."""
        try:
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                history = []
                messages = self.memory.chat_memory.messages
                
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        human_msg = messages[i]
                        ai_msg = messages[i + 1]
                        history.append({
                            "question": human_msg.content,
                            "answer": ai_msg.content,
                            "timestamp": datetime.now().isoformat()  # Approximate
                        })
                
                return history
            return []
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

def main():
    """CLI entry point for testing the chat system."""
    Config.setup_logging()
    
    try:
        print("🤖 Initializing Personal RAG Chat System...")
        print("📋 Loading configuration...")
        
        # Create config instance to check provider before full initialization
        config = Config()
        provider = config.LLM_PROVIDER.lower()
        
        if provider in ["mistral", "ollama"]:
            print("🏠 Using local Mistral model via Ollama")
            print("💡 Make sure Ollama is running and 'mistral:7b' model is installed")
        elif provider == "gemini":
            print("☁️ Using Google Gemini API")
            print("🔑 Using API key from environment variables")
        
        # Initialize the chat system
        chat = PersonalRAGChat()
        
        print("✅ Chat system initialized successfully!")
        print(f"🎯 Active model: {provider.upper()}")
        
        # Interactive chat loop
        print("\n💬 You can now ask questions about your documents.")
        print("Type 'quit' to exit, 'clear' to clear history, 'history' to see conversation history.")
        print("Type 'status' to see current model information.\n")
        
        while True:
            try:
                question = input("❓ Ask a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif question.lower() == 'clear':
                    chat.clear_history()
                    print("🧹 Conversation history cleared!")
                    continue
                
                elif question.lower() == 'status':
                    provider = chat.config.LLM_PROVIDER.lower()
                    print(f"\n📊 System Status:")
                    print(f"   🤖 LLM Provider: {provider.upper()}")
                    if provider == "gemini":
                        model_name = chat.config.GEMINI_MODEL or "gemini-2.5-flash"
                        print(f"   📱 Model: {model_name}")
                        print(f"   🔑 API Key: {'✅ Set' if chat.config.GEMINI_API_KEY else '❌ Missing'}")
                    elif provider in ["mistral", "ollama"]:
                        print(f"   📱 Model: mistral:7b")
                        print(f"   🏠 Type: Local (Ollama)")
                    
                    history_count = len(chat.get_conversation_history())
                    print(f"   💭 Conversation History: {history_count} exchanges")
                    continue
                
                elif question.lower() == 'history':
                    history = chat.get_conversation_history()
                    if history:
                        print("\n📖 Conversation History:")
                        for i, exchange in enumerate(history, 1):
                            print(f"\n{i}. Q: {exchange['question']}")
                            print(f"   A: {exchange['answer'][:200]}...")
                    else:
                        print("📝 No conversation history yet.")
                    continue
                
                elif not question:
                    continue
                
                print("\n🔍 Searching documents...")
                result = chat.ask_question(question)
                
                print(f"\n🤖 Answer: {result['answer']}")
                
                if 'sources' in result and result['sources']:
                    print(f"\n📚 Sources ({result['num_sources']} documents):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['file_name']}")
                        print(f"   Preview: {source['content_preview'][:150]}...")
                
                print(f"\n⏰ Answered at: {result['timestamp']}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"❌ Error: {e}")
    
    except Exception as e:
        logger.error(f"Chat initialization failed: {e}")
        print(f"❌ Chat initialization failed: {e}")
        
        # Provide helpful error messages based on common issues
        if "ollama" in str(e).lower() or "connection" in str(e).lower():
            print("\n💡 Troubleshooting tips:")
            print("   1. Make sure Ollama is installed and running")
            print("   2. Install Mistral model: ollama pull mistral:7b")
            print("   3. Check if Ollama service is running: ollama list")
        elif "api" in str(e).lower() or "key" in str(e).lower():
            print("\n💡 Troubleshooting tips:")
            print("   1. Check if GEMINI_API_KEY environment variable is set")
            print("   2. Verify your Google AI API key is valid")
            print("   3. Make sure you have credits/quota available")


if __name__ == "__main__":
    main()