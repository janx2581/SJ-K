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
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import openai
import os
from langchain.text_splitter import CharacterTextSplitter

# Set up API key
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["api_key"] = st.secrets["OPENAI_API_KEY"]

# Load and split the text file
txt_file_path = 'SJ-K.txt' #My thesis
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding=embeddings)

# Create conversation chain
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
# llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")  # Uncomment if using GPT-4
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
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
st.title("Tasha er sej")
st.markdown("""
Welcome to the Thesis Assistant for SJ-K. This tool is designed to help you quickly understand the key points from the thesis without needing to read the entire document. Simply click on a predefined question or enter your own to get started.
""")

# Predefined questions
queries = {
    "What is SJ&K?": "What is SJ&K?",
    "Hvad er konklusionen?": "Hvad er konklusionen på specialet?",
    "What is the status of Danish politics?": "What is the status of Danish politics?",
    "How can SJ&K help me understand the status of Danish politics?": "How can SJ&K help me understand the status of Danish politics?",
}

st.markdown("""
### Select a Question
Click on a question to get an answer, or type your own question in the input box below.
""")

# Initialize a variable to hold the selected query
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None

# Create buttons for each predefined query and handle selection
for button_label, query in queries.items():
    if st.button(button_label):
        st.session_state.selected_query = query
        break

# User input
user_query = st.text_input("Ask your questions here:")

if user_query:
    st.session_state.selected_query = user_query

# Check if a query has been selected
if st.session_state.selected_query:
    answer = get_answer(st.session_state.selected_query)
    st.write(f"**Query:** {st.session_state.selected_query}")
    st.write("**Answer:**")
    st.write(answer)

# Function to clear selection
def clear_selection():
    st.session_state.selected_query = None

st.markdown("____________________")

if st.button("Clear"):
    clear_selection()

# Read the contents of the file
with open(txt_file_path, 'r', encoding='utf-8') as file:
    sjk_text = file.read()

# Dropdown (expander) for displaying non-editable information
with st.expander("Transparency: See the information used in the model here:"):
    st.write(sjk_text)
