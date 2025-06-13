import os
import streamlit as st
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# ✅ Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAeWl6Vz7Mc3YIkZyDr3LdSpRHaepmsbOM"

# ✅ Streamlit UI
st.set_page_config(page_title="RAG Chat with Memory", layout="wide")
st.title("📚 Chat with PDF + Memory using Gemini")

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf is not None:
    with st.spinner("Processing PDF..."):
        # ✅ Save and load PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # ✅ Split PDF text
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # ✅ Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(chunks, embedding=embeddings)
        retriever = vectordb.as_retriever()

        # ✅ Memory and LLM setup
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

        # ✅ Create RAG chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        # ✅ Handle chat UI
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask a question from your PDF:")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke({"question": user_input})
                    st.session_state.chat_history.append((user_input, response["answer"]))
                except Exception as e:
                    st.error(f"Error during response: {e}")

        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
