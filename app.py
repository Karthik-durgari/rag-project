import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

st.set_page_config(page_title="ECE Research Assistant", layout="wide")

st.title("📘 ECE Research Assistant 🚀")

# ✅ Get API key from Streamlit Cloud secrets / local env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY not found in environment variables")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# -------------------------------
# 🔥 Initialize embedding model (cache for speed)
# -------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:

    # Save PDF temporarily
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

    st.success(f"PDF loaded ✔ | Chunks created: {len(chunks)}")

    # Vector DB (cached per session)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    query = st.text_input("💬 Ask a question from your PDF")

    if query:

        # Retrieve top chunks
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # ✅ FIXED GEMINI MODEL (stable version)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )

        prompt = f"""
You are an intelligent assistant.
Answer ONLY using the given context.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

        st.subheader("🧠 Answer")
        st.write(response.content)