import streamlit as st
import google.generativeai as genai
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# 🔐 API key for Gemini 2.0 (1.5 Flash)
GOOGLE_API_KEY = "AIzaSyDQaU4Q-8iY-swfqaL2cGAB1-EXpjlvvM8"
genai.configure(api_key=GOOGLE_API_KEY)

# 🌟 Use Gemini 1.5 Flash model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# 🔍 DuckDuckGo Tool via LangChain
search = DuckDuckGoSearchAPIWrapper()

search_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for answering questions about current events or factual real-world knowledge."
)

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="⚡ Gemini Flash Q&A", page_icon="⚡")
st.title("⚡ Real-Time Q&A with Gemini 2.0 Flash")
st.markdown("Ask anything about the world 🌍 — powered by Gemini Flash and DuckDuckGo!")

# 💬 User input
question = st.text_input("💬 What's your question?")
ask_button = st.button("🚀 Get Answer")

if ask_button and question:
    try:
        with st.spinner("Thinking with Gemini Flash..."):
            # First try tool
            tool_answer = search_tool.run(question)

            # Feed tool result to Gemini for better reasoning
            prompt = f"""Answer the question based on this web search result:
            \nSearch Result: {tool_answer}
            \nQuestion: {question}
            """
            response = model.generate_content(prompt)

        st.success("✅ Here's what I found:")
        st.write(response.text)
    except Exception as e:
        st.error("❌ Something went wrong.")
        st.code(str(e), language="bash")
