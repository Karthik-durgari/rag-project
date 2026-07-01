import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- UI ----------------
st.set_page_config(page_title="ECE RAG Assistant", layout="wide")
st.title("📘 ECE Research Assistant 🚀")

# ---------------- API KEY ----------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables")
    st.stop()

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FORCE RESET FUNCTION ----------------
def reset_all():
    st.session_state.vectorstore = None
    st.session_state.chat_history = []

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    # 🔥 HARD RESET ON NEW FILE
    reset_all()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    st.success(f"✅ PDF Loaded | Chunks created: {len(chunks)}")
    st.info("Now you can ask questions below 👇")

    # 🔥 NEW VECTORSTORE EACH TIME
    st.session_state.vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

# ---------------- QUERY ----------------
query = st.text_input("Ask a question from your PDF")

# ---------------- CHAT HISTORY ----------------
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for role, msg in st.session_state.chat_history:
        st.write(f"**{role}:** {msg}")

# ---------------- RAG FLOW ----------------
if query:

    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF first 📄")
        st.stop()

    docs = st.session_state.vectorstore.similarity_search(query, k=4)

    if not docs:
        st.warning("No relevant content found in this PDF")
        st.stop()

    context = "\n\n".join([d.page_content for d in docs])

    # 🔥 STABLE MODEL (DO NOT CHANGE)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = f"""
You are an expert assistant.

Answer ONLY using the context below.
If answer is not in context, say "Not found in document".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("Assistant", answer))

    st.subheader("🧠 Answer")
    st.write(answer)