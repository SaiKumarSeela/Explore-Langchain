from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms  import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. please  response to the user queries."),
        ("user", "Question: {question}")

    ]

)

# Streamlit framework
st.title("Langchain Demo with Ollama (using Llama3.1)")
input_text = st.text_input("Search the topic u want")

# ollama - llama3.1
llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

