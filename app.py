import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Check for Groq API key
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions as accurately as possible:
    Question: {input}
    """
)

# Streamlit App UI
st.set_page_config(page_title="Groq LLM Q&A", page_icon="✨", layout="centered")

# Custom CSS for enhanced UI with better contrast
st.markdown(
    """
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(120deg, #ffffff, #f0f4ff);
            color: #333333;
        }
        .main-header {
            background-color: #4a90e2;
            color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            margin: 0;
            font-size: 38px;
        }
        .main-header p {
            margin: 5px 0 0;
            font-size: 18px;
        }
        .input-section {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 2px solid #e0e6ed;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: #333333;
        }
        .response-section {
            background-color: #f9fcff;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            border: 2px solid #d6e4ff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: #333333;
        }
        .response-section h3 {
            color: #2c3e50;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #636e72;
        }
        .footer a {
            color: #4a90e2;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .stButton>button {
            background: linear-gradient(45deg, #6a11cb, #2575fc);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 18px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #5b0dbf, #1959d1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>Groq LLM Q&A ✨</h1>
        <p>Get instant, accurate answers powered by Groq AI!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown(
    """
    <div class="input-section">
        <label style="font-size:18px; font-weight:bold;">Enter your question:</label>
    </div>
    """,
    unsafe_allow_html=True,
)
user_query = st.text_input("", key="user_query", placeholder="Type your question here...")

# Button to trigger response generation
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Format the input prompt
                formatted_prompt = prompt.format_messages(input=user_query)

                # Pass the formatted prompt to the model
                response = llm.invoke(formatted_prompt)

                # Display response
                if isinstance(response, AIMessage):
                    st.markdown(
                        f"""
                        <div class="response-section">
                            <h3>Answer:</h3>
                            <p>{response.content}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("Unexpected response format from the model.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter a valid question!")

# Footer
st.markdown(
    """
    <div class="footer">
        Built with 💻 by <a href="#">Arif Ahmad Khan</a>
    </div>
    """,
    unsafe_allow_html=True,
)
