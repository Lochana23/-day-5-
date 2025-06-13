import os
import streamlit as st

# LangChain core and modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage

# LangChain integrations
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# ✅ Set Gemini API Key (for demo only - use secure method in production)
GOOGLE_API_KEY = "AIzaSyAeWl6Vz7Mc3YIkZyDr3LdSpRHaepmsbOM"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ✅ Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# ✅ Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("user", "Translate this sentence to French:\n{input}")
])

# ✅ LLM Chain using Runnable
chain: Runnable = prompt | llm

# ✅ Setup memory
memory = ConversationBufferMemory(return_messages=True)

# ✅ Streamlit UI
st.set_page_config(page_title="Gemini Translator", layout="centered")
st.title("🌐 English ➡️ French Translator with Gemini")

# ✅ Input field
input_text = st.text_input("Enter an English sentence:")

# ✅ Translate button
if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter a sentence.")
    else:
        try:
            response = chain.invoke({"input": input_text})
            translation = response.content if isinstance(response, AIMessage) else str(response)
            st.success("✅ Translation complete!")
            st.text_area("French Translation:", value=translation, height=150)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# ✅ Optional Memory Chat
if st.checkbox("Enable Memory Chat"):
    conversation = ConversationChain(llm=llm, memory=memory)
    try:
        mem_response = conversation.predict(input=input_text)
        st.info("🧠 Memory Response:")
        st.write(mem_response)
    except Exception as e:
        st.error(f"Memory error: {e}")

# ✅ Optional Simple RAG
if st.checkbox("Enable Simple RAG"):
    try:
        documents = ["Paris is the capital of France. It is known for the Eiffel Tower."]

        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        split_docs = splitter.create_documents(documents)

        # ✅ Embeddings using Hugging Face MiniLM (lightweight)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # RAG logic
        doc_chain = StuffDocumentsChain(llm=llm, prompt=prompt)
        relevant_docs = retriever.get_relevant_documents(input_text)
        rag_result = doc_chain.run(input_documents=relevant_docs, input=input_text)

        st.info("📚 RAG Output:")
        st.write(rag_result)
    except Exception as e:
        st.error(f"RAG error: {e}")
