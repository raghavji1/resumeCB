import streamlit as st
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders.pdf import PyPDFLoader
from dotenv import load_dotenv
from utils import *
from myvoice import *

load_dotenv()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to push embedded data to Vector Store - Pinecone
def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="qa", embedding=embeddings)
    return vector_store

# Function to get response based on user input
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

st.set_page_config(page_title="Find Your Fav Job", page_icon="🕵️‍♀️")

# session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello your CareerJunction is here how can I assist you today..?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()  

# Voice recorder sidebar
st.sidebar.title("Voice Recorder")
if st.sidebar.button("Record Voice"):
    listen()

# Conversation
st.header('CareerJunction 🚂 ')
st.text('Powerd by OpenAI')

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
