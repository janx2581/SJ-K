"""
from google.colab import drive
drive.mount('/content/drive/')
"""

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
import tiktoken
from langchain_openai import ChatOpenAI
# print("done importing")
import openai

import os


# openai_api_key = st.secrets["OPENAI_API_KEY"]
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["api_key"] = st.secrets["OPENAI_API_KEY"]


# Prompt the user for their OpenAI API key
# api_key = input("Please enter your OpenAI API key: ")

# Set the API key as an environment variable
# os.environ["OPENAI_API_KEY"] = api_key

# Optionally, check that the environment variable was set correctly
# print("OPENAI_API_KEY has been set!")


llm_model = "gpt-3.5-turbo"

from langchain.text_splitter import CharacterTextSplitter

# print("done importing the CharacterTextSplitter")


# txt_file_path = '/content/drive/MyDrive/scalexi.txt'
txt_file_path = 'SJ-K.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Data
# print("data")
# data


# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding=embeddings)

# Create conversation chain
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
# llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)


def get_answer(query):
    result = conversation_chain({"question": query})
    return result["answer"]

# Streamlit app layout
st.title("SJ-K RAG model")
st.markdown("""
En demo for min ragmodel.  

Den er sat til at se på data fra SJ-K.dk, men meningen med tiden er naturligvis at sætte den til at se på data fra folketinget.
""", unsafe_allow_html=False)

# Define your queries here
queries = {
    "What is SJ&K?": "What is SJ&K?",
    "How can they help me?": "How can they help me?",
    "What is the status of Danish politics?*": "What is the status of Danish politics?",
    "How can SJ&K help me understand the status of Danish politics?": "How can SJ&K help me understand the status of Danish politics?",
}

# Streamlit app layout continued
st.markdown("""
Tryk på et spørgsmål eller skriv dit eget spørgsmål i chatten nedenfor.
""", unsafe_allow_html=False)

# Initialize session state for selected and user-entered query
if 'selected_query' not in st.session_state:
    st.session_state['selected_query'] = None
if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ""

# Create buttons for each query
for button_label, query in queries.items():
    if st.button(button_label):
        st.session_state.selected_query = query
        st.session_state.user_query = ""  # Reset user input when a button is pressed

# Display the result if a predefined query was selected
if st.session_state.selected_query and not st.session_state.user_query:
    answer = get_answer(st.session_state.selected_query)
    st.write(f"Query: {st.session_state.selected_query}")
    st.write("Answer:")
    st.write(answer)

# User input
st.session_state.user_query = st.text_input("Ask your questions here (Preferably in English men dansk virker også):", value=st.session_state.user_query)

if st.session_state.user_query and not st.session_state.selected_query:
    # Get answer from the conversational chain
    answer = get_answer(st.session_state.user_query)

    # Display the answer
    st.write("Query:")
    st.write(st.session_state.user_query)
    st.write("Answer:")
    st.write(answer)

# Function to clear selection and user query
def clear_selection():
    st.session_state.selected_query = None
    st.session_state.user_query = ""

st.markdown("____________________")

# Clear button
if st.button("Clear chat (måske skal I trykke to gange før det virker)"):
    clear_selection()

# Reading and displaying file content
with open(txt_file_path, 'r', encoding='utf-8') as file:
    sjk_text = file.read()

with st.expander("For transparens: Se hvilken information jeg har copy/pasted fra sj-k.dk ind i modellen her:"):
    st.write(sjk_text)

st.markdown("*Obs: Jeg har ikke taget noget om nyhedsbreve, media eller buzzed med.*")
st.markdown("*Asterix: Jeg har inkluderet dette spørgsmål for at vise, at den ikke bare finder på noget tilfældigt*")



