from pathlib import Path
import streamlit as st
import logging
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.chat import PersonalRAGChat
from src.ingest import DocumentIngestor
from src.retreiver import DocumentRetriever
from src.config import Config

# ---------------------------
# Setup Logging
# ---------------------------
Config.setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Personal RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Personal RAG Chatbot")
st.markdown("Ask questions about your personal documents!")

# ---------------------------
# Session State Init
# ---------------------------
if 'chat' not in st.session_state:
    try:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chat = PersonalRAGChat()
        st.success("✅ Chatbot initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("📊 Show Statistics"):
        try:
            retriever = DocumentRetriever()
            st.json(retriever.get_vectorstore_stats())
        except Exception as e:
            st.error(f"Error getting stats: {e}")

    if st.button("🧹 Clear History"):
        st.session_state.chat.clear_history()
        st.session_state.messages = []
        st.success("History cleared!")
        st.rerun()

    if st.button("🔄 Re-ingest Documents"):
        try:
            with st.spinner("Re-ingesting documents..."):
                DocumentIngestor().ingest_documents()
                st.session_state.chat = PersonalRAGChat()
            st.success("Documents re-ingested successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error re-ingesting documents: {e}")

# ---------------------------
# Chat Display
# ---------------------------
st.subheader("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------
# Chat Input
# ---------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chat.ask_question(prompt, include_sources=True)
                if result.get("error"):
                    st.error(f"Error: {result['answer']}")
                else:
                    st.write(result["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
