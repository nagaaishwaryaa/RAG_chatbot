# rag_chatbot_faiss.py

import os
from tempfile import NamedTemporaryFile
import streamlit as st
from dotenv import load_dotenv

# LangChain v0.2+ compliant imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file.")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot (FAISS + GPT-4o-mini)", layout="wide")
st.title("📄💬 Chat with your PDF (FAISS + GPT-4o-mini)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
VECTORSTORE_PATH = "faiss_index"

def process_pdf_and_create_chain(file):
    try:
        # Save uploaded PDF temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            st.error("⚠️ PDF is empty or unreadable.")
            return None

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        # Check if FAISS already exists
        if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
            vectordb = FAISS.load_local(VECTORSTORE_PATH, embedding, allow_dangerous_deserialization=True)
        else:
            vectordb = FAISS.from_documents(chunks, embedding)
            vectordb.save_local(VECTORSTORE_PATH)

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return rag_chain

    except Exception as e:
        st.error(f"❌ Error during PDF processing: {str(e)}")
        return None

rag_chain = None
if uploaded_file:
    with st.spinner("Processing PDF and indexing..."):
        rag_chain = process_pdf_and_create_chain(uploaded_file)

# Chat interface
if rag_chain:
    user_input = st.chat_input("Ask a question from your PDF...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.run(user_input)
                st.session_state.chat_history.append(("bot", response))
            except Exception as e:
                st.session_state.chat_history.append(("bot", f"❌ Error: {str(e)}"))

# Display conversation
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
