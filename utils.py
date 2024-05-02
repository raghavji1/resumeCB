from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pypdf import PdfReader
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS  

import uuid

#Creating session variables
if 'unique_id' not in st.session_state: 
    st.session_state['unique_id'] =''   


#map reduce chain for the document summary
list = []

def document_response(docs):
        prompt_template = """you have some resumes and you have to provide only the name of candidate, and  experience of each candidate in years, and by skills you have to fetch his category like a software developer, ai developer, manager etc of the given :
        "{text}"
        do not show full document and seperate them by comma like 'candidate name' , 'experiencxe', 'category' :"""
        prompt = PromptTemplate.from_template(prompt_template)  
        llm=ChatOpenAI(temperature=0)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        list.append(stuff_chain.run(docs))
        st.sidebar.write(list)
        return list

       

# Assuming this function encodes the question into a vector representation
def encode_question(question,embeddings):
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector

def save_vector_store(text_chunks,embeddings):

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db = FAISS.load_local("faiss_index_V2", embeddings, allow_dangerous_deserialization=True)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index_V2')

    return st.write("vector Store is Saved")


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs





def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
        )
    # create a vectorstore from the chunks
    vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation with the user name and his category which is define in the list,  generate a search query to look up in order to get information relevant to the conversation")
    ])
    # If there is no chat_history, then the input is just passed directly to the retriever. 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

#function for uploading resume and extract their candidate name, their experience, and their category of skills
def upload_resume():
    pdf = st.sidebar.file_uploader("ok Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)
    submit=st.sidebar.button("Help me with the analysis")
    if submit:
                st.session_state['unique_id']=uuid.uuid4().hex
                final_docs_list=create_docs(pdf,st.session_state['unique_id'])
                st.sidebar.write("*Resumes uploaded* :"+str(len(final_docs_list)))
                document_response(final_docs_list)


def  generate_response_jobseker(user_input):
    if user_input.lower() == "job seeker":
        upload_resume()
        prompt=ChatPromptTemplate.from_messages([
      ("system", "you are an AI Chatbot who first ask user that which language he will prefer for example HINDI, ENGLISH, or HINGLISH(if hinglish-for exmaple, namaste me aapke liye kya kr skta hu  ) for communication. than you will ask him options like who he is JOB SEEKE, when they enter their identity , if the user is 'Job Seaker' than you will ask him in which sector he want to do job, you will find the skills/ if he is software based than you will find the languages what he know in his resume what he will upload ,and ask them to upload their students resume:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    return prompt

def recruiter(user_input):
    if user_input.lower() == "recruiter":
        prompt=ChatPromptTemplate.from_messages([
    ("system", "you are an AI Chatbot who first ask user that which language he will prefer for example HINDI, ENGLISH, or HINGLISH(if hinglish-for exmaple, namaste me aapke liye kya kr skta hu  ) for communication. than you will ask him options like who he is JOB SEEKER, RECRUTERM or INSTITUTE, when they enter their identity , if he is 'recruiter' than you will ask him in what sector he want to hire, you will ask him the prefer language and the location than we will provide the number of the resume and the names in the resumes what we have which have matching the requirements of the recriter :\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    return prompt

def institute(user_input):
    if user_input.lower() == "institute":
        upload_resume()
        prompt=ChatPromptTemplate.from_messages([
        ("system", "you are an AI Chatbot who first ask user that which language he will prefer for example HINDI, ENGLISH, or HINGLISH(if hinglish-for exmaple, namaste me aapke liye kya kr skta hu  ) for communication. than you will ask him options like who he is JOB SEEKER, RECRUTERM or INSTITUTE, when they enter their identity , if he is 'institue' than you will ask how many students do you have for recruitement process and then you will ask him to upload resumes accourdingly than we will suggest them the job openings as per their students details and resumes we will provide the information according to the name of student and the simplar job profile of compay which we have :\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ])
    return prompt



def handle_identity_input(user_input):
    if user_input.lower() == "job seeker":
        generate_response_jobseker(user_input)
                    
    elif  user_input.lower() == "recruiter":
        recruiter(user_input)
        
    elif  user_input.lower() == "institute":
        institute(user_input)
        


    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", """you are an AI Chatbot who first ask user that which language he will prefer for example HINDI, ENGLISH, or HINGLISH(if hinglish-for exmaple, namaste me aapke liye kya kr skta hu  ) for communication. than you will ask him options like who he is JOB SEEKER, RECRUTERM or INSTITUTE, when they enter their identity , if the user is 'Job Seaker' than you will ask him in which sector he want to do job, if the person is 'recrutor' than you will ask in how many person do you want and for which role, if ' institude' you will ask them to how many students do you have and what are their major skills and ask them to upload their students resume
       you are a so smart ai model you not show him all the job opening directly you have to ask his all the expectation about the job opening and all like loaction, salary package, You will ask this in different questions one by one . field and field's knowing also you suggest him some similar type of jobs according to his capablities in his resume, you will give tips for the job interview and all after he select the job what he need,
       do not give him the job appling link you will show him link after he select the job when you suggest him some openings you will only present the company name, job role, location, number of openings in table form serial wise and after he select his job you will display it's appling link and suggestions for the interview and form fillings,
       you also find the emotion of the user as he changes his choises in different different sectors which he has no experties if he changes him mind again and again you will find him as frusted, confused and so on you will give him some suggestions while asking his capablities in differet questions :\n\n{context}"""),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


