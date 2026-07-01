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
    st.error("❌ GOOGLE_API_KEY not found in Streamlit Secrets")
    st.stop()

# ---------------- EMBEDDINGS (cached) ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    st.success(f"✅ PDF Loaded | Chunks created: {len(chunks)}")

    # Vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # ---------------- USER QUERY ----------------
query = st.text_input("Ask a question from your PDF")

if query:

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )

    # Prompt
    prompt = f"""
You are an expert assistant.
Answer ONLY using the given context.

Context:
{context}

Question:
{query}
"""

    # Get response
    response = llm.invoke(prompt)

    # Output safely
    st.subheader("🧠 Answer")

    try:
        if isinstance(response.content, list):
            st.write(response.content[0]["text"])
        else:
            st.write(response.content)
    except:
        st.write(str(response))