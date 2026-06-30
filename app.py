import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

st.title("ECE Research Assistant 🚀")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    # Save PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    st.success("RAG system ready 🚀")

    query = st.text_input("Ask a question from your PDF")

    if query:

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

        # Prompt
        prompt = f"""
        You are a helpful assistant.
        Answer ONLY using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.subheader("Answer 🧠")
        st.write(response.content)