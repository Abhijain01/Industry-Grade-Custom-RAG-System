import streamlit as st
import asyncio
import os
import tempfile
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import warnings
import ssl

# Bypass SSL verification for local dev/proxies
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")

load_dotenv()

# Must be imported after setting up event loop in Streamlit usually, but fine here
from src.pipeline import RAGPipeline

# --- Page Config ---
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container .main .block-container { max-width: 1000px; }
    .stChatFloatingInputContainer { bottom: 20px; }
    .citation { font-size: 0.8em; color: gray; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "pipeline" not in st.session_state:
    try:
        # Initialize pipeline without triggering ingestion again
        st.session_state.pipeline = RAGPipeline("config.yaml")
        st.toast("RAG Pipeline Loaded Successfully!", icon="✅")
    except Exception as e:
        st.error(f"Failed to load RAG Pipeline: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main App Execution ---
def main():
    st.title("🧠 Enterprise RAG Assistant")
    st.caption("A custom-built Retrieval-Augmented Generation system using FAISS, BM25, and Cross-Encoders.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Query Mode Selection
        query_mode = st.selectbox(
            "Query Mode",
            ["standard", "hyde", "multi_query"],
            index=0,
            help="standard: Direct search. hyde: LLM hallucinates an answer to search. multi_query: LLM breaks query into 5 sub-queries."
        )
        
        top_k = st.slider("Top K Retrieval", min_value=1, max_value=20, value=5)
        
        st.divider()
        st.header("📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["pdf", "txt", "docx", "md"], 
            accept_multiple_files=True
        )
        
        if st.button("Ingest Uploaded Files", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload files first.")
            else:
                with st.spinner("Processing documents..."):
                    # Create a temporary directory to save uploaded files for ingestion
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for uf in uploaded_files:
                            temp_path = Path(temp_dir) / uf.name
                            with open(temp_path, "wb") as f:
                                f.write(uf.getbuffer())
                                
                        # Run ingestion using asyncio
                        async def run_ingestion():
                            await st.session_state.pipeline.ingest_directory(temp_dir)
                            
                        asyncio.run(run_ingestion())
                        
                    st.success(f"Successfully ingested {len(uploaded_files)} files!")
                    st.balloons()
                    
        st.divider()
        st.caption("Powered by locally hosted FAISS & Sentence-Transformers.")

    # --- Chat Interface ---
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Context and Answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Retrieving context & generating answer..."):
                try:
                    # Async task wrapper
                    async def fetch_answer():
                        # We use stream=False here for simplicity in Streamlit, 
                        # but stream=True can be implemented using st.write_stream
                        response = await st.session_state.pipeline.ask(
                            query=prompt,
                            query_mode=query_mode,
                            top_k=top_k,
                            stream=False
                        )
                        return response
                    
                    full_response = asyncio.run(fetch_answer())
                    
                    # Display response
                    message_placeholder.markdown(full_response)
                    
                    # Append to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.exception(e)

if __name__ == "__main__":
    main()
