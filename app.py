import streamlit as st
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from pypdf import PdfReader
from utils import *



import json
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to push embedded data to Vector Store - Pinecone

def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="qa",embedding=embeddings)
    return vector_store


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({

        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


# loader = PyPDFLoader('Remote_hiring.pdf')
# docs = loader.load()
# push_to_pinecone(PINECONE_API_KEY,"gcp-starter","qa",embeddings,docs)

# loader = CSVLoader(file_path="G:/VKAPS Internship Projects AI/New folder (2)/Components/Book1.csv")
# data = loader.load()
# push_to_pinecone(PINECONE_API_KEY,"gcp-starter","qa",embeddings,data)

# app config
st.set_page_config(page_title="Find Your Fav Job", page_icon="🕵️‍♀️")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello This is NaukariDhoondho Bot here how can I assist you today..?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()  


#upload resume function 

st.header('NaukariDhoondho ChatBot ')
st.text('Powerd by OpenAI')


# conversation~
user_query = st.chat_input("Ask your query here About the Given PDF...")
for message in st.session_state.chat_history :
    if isinstance(message, HumanMessage)  :
        with st.chat_message("You")   :
            st.markdown(message.content)
    else  :
        with st.chat_message("AI"):
            st.markdown(message.content)


if user_query:
    response = get_response(user_query)
    # Display user's question
    with st.chat_message("You"):
        
        st.markdown(user_query)
    # Display AI's answer
    with st.chat_message("AI"):
        st.markdown(response)

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
