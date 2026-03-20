"""
AI Research Assistant - Streamlit Frontend
Production-grade web interface for research assistance
"""

import streamlit as st
import requests
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .source-card {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-message-user {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .chat-message-assistant {
        background-color: #F5F5F5;
        border-left: 4px solid #757575;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_documents():
    """Get list of indexed documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def upload_file(file):
    """Upload a file to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=files,
            timeout=60,
        )
        return response
    except Exception as e:
        return None


def upload_url(url):
    """Upload a URL to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/upload/url",
            params={"source_url": url},
            timeout=60,
        )
        return response
    except Exception as e:
        return None


def query_documents(query, top_k=5, search_type="hybrid"):
    """Query the knowledge base."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"query": query, "top_k": top_k, "search_type": search_type},
            timeout=120,
        )
        return response
    except Exception as e:
        return None


def query_documents_streaming(query, top_k=5, search_type="hybrid"):
    """Query the knowledge base with streaming."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query/stream",
            json={"query": query, "top_k": top_k, "search_type": search_type},
            stream=True,
            timeout=120,
        )
        return response
    except Exception as e:
        return None


def generate_summary(source=None, summary_type="short"):
    """Generate a summary."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/summary",
            json={"source": source, "summary_type": summary_type},
            timeout=120,
        )
        return response
    except Exception as e:
        return None


def delete_document(source):
    """Delete a document."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/documents",
            json={"source": source},
            timeout=30,
        )
        return response
    except Exception as e:
        return None


# Sidebar
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.markdown("---")

    # API Status
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.warning("Make sure the API server is running: `uvicorn app.api.main:app`")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📤 Upload", "🔍 Query", "📝 Summary", "📚 Knowledge Base"],
    )


# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Welcome to AI Research Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    This is an AI-powered research assistant that uses **Retrieval Augmented Generation (RAG)**
    to help you analyze documents and get accurate answers to your research questions.
    """)

    # Features
    st.markdown("### 🚀 Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📄 Document Ingestion**
        - Upload PDF, TXT, MD, HTML files
        - Fetch content from URLs
        - Automatic text extraction
        """)

    with col2:
        st.markdown("""
        **🔍 Semantic Search**
        - Vector-based similarity search
        - ChromaDB vector storage
        - Mistral embeddings
        """)

    with col3:
        st.markdown("""
        **💬 Question Answering**
        - RAG with Mistral AI
        - Source citations
        - Confidence scoring
        """)

    # Stats
    st.markdown("### 📊 System Statistics")
    stats = get_stats()

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Chunks", stats.get("total_chunks", 0))
        with col3:
            size_mb = stats.get("storage_size_bytes", 0) / (1024 * 1024)
            st.metric("Storage", f"{size_mb:.2f} MB")
        with col4:
            last_updated = stats.get("last_updated", "N/A")
            st.metric("Last Updated", last_updated[:10] if last_updated != "N/A" else "N/A")
    else:
        st.info("No statistics available. Upload some documents to get started!")

    # Quick start
    st.markdown("### 🚦 Quick Start")
    st.markdown("""
    1. **Upload Documents** - Go to the Upload page to add your research materials
    2. **Ask Questions** - Use the Query page to ask research questions
    3. **Generate Summaries** - Create summaries of your documents
    """)


# Upload Page
elif page == "📤 Upload":
    st.markdown('<div class="main-header">Upload Documents</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 File Upload", "🌐 URL Upload"])

    with tab1:
        st.markdown("### Upload Files")
        st.markdown("Supported formats: PDF, TXT, MD, HTML")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md", "html"],
        )

        if uploaded_file:
            st.info(f"Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("Upload Document", type="primary"):
                with st.spinner("Processing document..."):
                    response = upload_file(uploaded_file)

                    if response and response.status_code == 200:
                        result = response.json()
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ {result.get('message', 'Upload successful!')}
                            <br>
                            Chunks created: {result.get('chunk_count', 0)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-box">
                            ❌ Error uploading document
                        </div>
                        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Fetch from URL")
        url = st.text_input("Enter URL", placeholder="https://example.com/article")

        if url:
            if st.button("Fetch URL Content", type="primary"):
                with st.spinner("Fetching content..."):
                    response = upload_url(url)

                    if response and response.status_code == 200:
                        result = response.json()
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ {result.get('message', 'URL processed successfully!')}
                            <br>
                            Chunks created: {result.get('chunk_count', 0)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-box">
                            ❌ Error fetching URL
                        </div>
                        """, unsafe_allow_html=True)


# Query Page
elif page == "🔍 Query":
    st.markdown('<div class="main-header">Research Query</div>', unsafe_allow_html=True)

    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")

    # Query input
    query = st.text_area(
        "Enter your research question:",
        height=100,
        placeholder="What are the key findings in the documents about...",
    )

    # Settings
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        stream_response = st.checkbox("Stream answer", value=True)
    with col2:
        search_type = st.selectbox(
            "Search Strategy",
            ["hybrid", "semantic", "mmr"],
            index=0,
            format_func=lambda x: {
                "hybrid": "Hybrid Search",
                "semantic": "Semantic ONLY",
                "mmr": "Diverse (MMR)"
            }.get(x, x)
        )
    with col3:
        top_k = st.slider("Documents to retrieve", 1, 10, 5)

    # Query button
    if st.button("🔍 Get Answer", type="primary"):
        if query:
            if stream_response:
                with st.spinner("Retrieving sources..."):
                    full_response = query_documents(query, top_k, search_type)
                    
                if full_response and full_response.status_code == 200:
                    result = full_response.json()
                    st.markdown("### 💬 Answer")
                    
                    # Streaming placeholder
                    answer_placeholder = st.empty()
                    full_answer = ""
                    
                    # Perform the streaming call
                    with requests.post(
                        f"{API_BASE_URL}/api/query/stream",
                        json={"query": query, "top_k": top_k, "search_type": search_type},
                        stream=True,
                        timeout=120,
                    ) as r:
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_answer += chunk
                                answer_placeholder.markdown(full_answer + "▌")
                    
                    answer_placeholder.markdown(full_answer)

                    # Confidence
                    confidence = result.get("confidence_score", 0)
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                    # Sources
                    sources = result.get("sources", [])
                    if sources:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('source', 'Unknown')}"):
                                st.markdown(f"**Relevance Score:** {source.get('score', 0):.4f}")
                                st.markdown(f"**Content Preview:**")
                                st.markdown(source.get("content_preview", ""))

                    # Processing time
                    st.caption(f"Processed in {result.get('processing_time_seconds', 0):.2f}s")
                    
                    # Add to history
                    if query not in st.session_state.query_history:
                        st.session_state.query_history.append(query)
                else:
                    st.error("Error processing query. Please try again.")
            else:
                with st.spinner("Searching and generating answer..."):
                    response = query_documents(query, top_k, search_type)

                    if response and response.status_code == 200:
                        result = response.json()

                        # Display answer
                        st.markdown("### 💬 Answer")
                        st.markdown(result.get("answer", "No answer generated"))

                        # Confidence
                        confidence = result.get("confidence_score", 0)
                        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                        # Sources
                        sources = result.get("sources", [])
                        if sources:
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"Source {i}: {source.get('source', 'Unknown')}"):
                                    st.markdown(f"**Relevance Score:** {source.get('score', 0):.4f}")
                                    st.markdown(f"**Content Preview:**")
                                    st.markdown(source.get("content_preview", ""))

                        # Processing time
                        st.caption(f"Processed in {result.get('processing_time_seconds', 0):.2f}s")
                        
                        # Add to history
                        if query not in st.session_state.query_history:
                            st.session_state.query_history.append(query)
                    else:
                        st.error("Error processing query. Please try again.")
        else:
            st.warning("Please enter a question.")


# Summary Page
elif page == "📝 Summary":
    st.markdown('<div class="main-header">Generate Summary</div>', unsafe_allow_html=True)

    st.markdown("Create summaries of your indexed documents in various formats.")

    # Summary type
    summary_type = st.selectbox(
        "Summary Type",
        ["short", "bullet", "detailed", "executive"],
        format_func=lambda x: {
            "short": "Short (2-3 sentences)",
            "bullet": "Bullet Points",
            "detailed": "Detailed Summary",
            "executive": "Executive Summary",
        }.get(x, x),
    )

    # Source selection
    documents = get_documents()
    if documents:
        source_options = ["All Documents"] + [d["source"] for d in documents]
        source = st.selectbox(
            "Select Source",
            source_options,
            format_func=lambda x: x if x == "All Documents" else f"{x[:50]}...",
        )

        if source == "All Documents":
            source = None
    else:
        source = None
        st.info("No documents available. Upload some documents first!")

    # Generate button
    if st.button("📝 Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            response = generate_summary(source, summary_type)

            if response and response.status_code == 200:
                result = response.json()

                # Display summary
                st.markdown("### Summary")
                st.markdown(result.get("summary", "No summary generated"))

                # Key points
                key_points = result.get("key_points", [])
                if key_points:
                    st.markdown("### Key Points")
                    for point in key_points:
                        st.markdown(f"- {point}")

                # Word count and time
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Word Count", result.get("word_count", 0))
                with col2:
                    st.metric("Processing Time", f"{result.get('processing_time_seconds', 0):.2f}s")
            else:
                st.error("Error generating summary. Please try again.")


# Knowledge Base Page
elif page == "📚 Knowledge Base":
    st.markdown('<div class="main-header">Knowledge Base</div>', unsafe_allow_html=True)

    st.markdown("View and manage your indexed documents.")

    # Get documents
    documents = get_documents()

    if documents:
        st.markdown(f"### 📚 {len(documents)} Document(s) Indexed")

        for doc in documents:
            with st.expander(f"📄 {doc.get('title', doc.get('source', 'Unknown'))}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                    st.markdown(f"**Type:** {doc.get('source_type', 'N/A')}")

                with col2:
                    st.markdown(f"**Chunks:** {doc.get('chunk_count', 0)}")
                    st.markdown(f"**Added:** {doc.get('created_at', 'N/A')[:10]}")

                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_{doc.get('source')}"):
                    response = delete_document(doc.get("source"))
                    if response and response.status_code == 200:
                        st.success("Document deleted!")
                        st.rerun()
                    else:
                        st.error("Error deleting document")

        # Stats
        st.markdown("---")
        st.markdown("### 📊 Storage Statistics")

        stats = get_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            with col3:
                size_mb = stats.get("storage_size_bytes", 0) / (1024 * 1024)
                st.metric("Storage Used", f"{size_mb:.2f} MB")
    else:
        st.info("No documents in the knowledge base. Upload some documents to get started!")


# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #757575;'>
        <p>AI Research Assistant v1.0.0 | Powered by LangChain & Mistral AI</p>
        <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
