
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
#print("done importing")
     

import os


#openai_api_key = st.secrets["OPENAI_API_KEY"]
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["api_key"] = st.secrets["OPENAI_API_KEY"]


# Prompt the user for their OpenAI API key
#api_key = input("Please enter your OpenAI API key: ")

# Set the API key as an environment variable
#os.environ["OPENAI_API_KEY"] = api_key

# Optionally, check that the environment variable was set correctly
#print("OPENAI_API_KEY has been set!")


llm_model = "gpt-3.5-turbo"
     

from langchain.text_splitter import CharacterTextSplitter
#print("done importing the CharacterTextSplitter")
     

#txt_file_path = '/content/drive/MyDrive/scalexi.txt'
txt_file_path = '/janx2581/SJ-K/SJ-K.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)
     

#Data
#print("data")
#data
     

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding=embeddings)

     

# Create conversation chain
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
#llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
memory = ConversationBufferMemory(
memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
        )




"""
query = "What is the status of Danish Politics?"
result = conversation_chain({"question": query})
answer = result["answer"]
answer
     
query = "How can Schultz Jørgensen & Kom help me understand the status of Danish politics?"
result = conversation_chain({"question": query})
answer = result["answer"]
answer
     
"""



st.title("Conversational AI")

# Session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User message input
user_message = st.text_input("You:", key="user_input")

# Function to handle conversation
def handle_conversation(user_message):
    if user_message:
        # Append user message to chat history
        st.session_state.chat_history.append(f"You: {user_message}")
        
        # Here, integrate your conversational AI logic
        # For demonstration, we're just echoing the user message
        # In practice, you'd use conversation_chain to get a response
        ai_response = f"AI: {user_message[::-1]}"  # Placeholder response logic
        
        # Append AI response to chat history
        st.session_state.chat_history.append(ai_response)
        
        # Reset user input
        st.session_state.user_input = ""

# Button to send message
if st.button("Send"):
    handle_conversation(user_message)

# Display chat history
for message in st.session_state.chat_history:
    st.text(message)
