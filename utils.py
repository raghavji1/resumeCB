import uuid
import streamlit as st
from pypdf import PdfReader
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import Pinecone as PineconeStore, FAISS


# Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''


# map reduce chain for the document summary
list = []


def document_response(docs):
    prompt_template = """you have some resumes and you have to provide only the name of candidate, and  experience of each candidate in years, and by skills you have to fetch his category like a software developer, ai developer, manager etc of the given :
    "{text}"
    do not show full document and seperate them by comma like 'candidate name' , 'experience', 'category' :"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    list.append(stuff_chain.run(docs))
    st.sidebar.write(list)
    return list


def encode_question(question, embeddings):
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector


def save_vector_store(text_chunks, embeddings):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db = FAISS.load_local("faiss_index_V2", embeddings, allow_dangerous_deserialization=True)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index_V2')
    return st.write("vector Store is Saved")


@st.cache_data
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


@st.cache_data
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        # Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": filename.file_id, "type=": filename.type, "size": filename.size,
                      "unique_id": unique_id},
        ))

    return docs


@st.cache_data
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(docs)
    pinecone = Pinecone(
        api_key=pinecone_apikey, environment=pinecone_environment
    )
    # create a vectorstore from the chunks
    vector_store = PineconeStore.from_documents(document_chunks, embeddings, index_name=pinecone_index_name)


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


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """you are a chatbot your name is 'NaukariDhoondho bot' your task is to intract with the coustomer and help him by multiple questions and answering way you will question the user about his all needs step by step and answer him accordingly,
        you can intract with user in very femiliar and simple form first you will great him and ask him in which language he will comfortable you can answer him accoring to his choise also his technique of questining and answering you can answer him in many languages but the most you will anser the user in three different language first ask him in what language you are comfortable for Ex. ENGLISH, HINDI(like- हमे आपसे बात करके ख़ुशी हुई ), HINGLISH(mix of hindi and english like- hume aapse baat krke khushi hui),
        after the language selection you will ask him about his identity  as a 'Job seeker' or 'Recruiter' or 'Institute' and then follow up with the conversation from there,
        If he is 'job seeker ' first you will ask him about in which field you are seraching the job and what expertise he has in this particular job profile each in differnt questions like if he is software developer as him anbout what languages you have experties, after that you will ask him some job related questions like location , salary package etc in on by one differnt questions, we will show upto 5 job opeings at a time in the table formate like company-name Position Number-of-Openings Loaction serial wise, we will share him link after he select the one also show him some suggestion about the job and the interview process of the particular job ,
        IF he is 'Recruiter' ask him about his company name and number of opeings  in the city and in which positions he is hiring and if they want the experiense person or a fresher and if they need person with number of opeings and many different questions accordingly,
        If he is 'Institute' ask him about his institute name location which type of person and technology do you have and if he want you will ask him about the number of students he have in his institute and you will ask to upload the resume of the particular student in his institute and in which catagory they have skills you will find out by diffrent questions and answering and suggest them the jobs opeings,
        The user wants the chatbot to leverage its NLP capabilities to analyze queries and conversation history, identifying keywords, phrases, or patterns indicating specific needs or challenges. Then, it should provide timely information such as alerts about job openings matching the user's profile, reminders for application deadlines or exams, links to helpful resources like interview tips, and guidance on navigating complex sections of online application forms (we will find user sentiment by his way to question and answring if he using some abusing words or changes his mind again and again we will make a cool down him and again make qustions about his priority with emogies),
        The chatbot offers personalized proactive assistance based on the user's profile and conversation history. It provides field-specific tips, suggests skill enhancement opportunities, shares job alerts, offers exam preparation resources, and assists with application processes,
        The chatbot uses NLP techniques to analyze user queries and conversation history, detecting emotions like frustration, confusion, or excitement. It may optionally integrate audio analysis tools to detect emotions from vocal cues. In response, the chatbot is trained to offer empathetic support, such as breaking down complex processes step-by-step or providing helpful resources in English/Hindi,
        If the user is satisfy with your results and he thanking us you will always reply him with emogies and always take feedback in the stars out of 5
        
        :\n\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# function for uploading resume and extract their candidate name, their experience, and their category of skills
def upload_resume():
    pdf = st.sidebar.file_uploader("ok Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
    submit = st.sidebar.button("Help me with the analysis")
    if submit:
        st.session_state['unique_id'] = uuid.uuid4().hex
        final_docs_list = create_docs(pdf, st.session_state['unique_id'])
        st.sidebar.write("*Resumes uploaded*: " + str(len(final_docs_list)))
        document_response(final_docs_list)
