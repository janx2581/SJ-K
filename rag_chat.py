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
# llm = ChatOpenAI(temperature=0.7, model_name="gpt-4") # if important for performance
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)



def get_answer(query):
    result = conversation_chain({"question": query})
    return result['answer'], result['chat_history']

# Streamlit app setup
st.set_page_config(page_title="Conversational QA System", layout="wide")
st.title("Conversational QA System")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

query = st.text_input("Enter your question:")

if query:
    answer, chat_history = get_answer(query)
    st.session_state.conversation_history.append((query, answer))

    st.subheader("Conversation History")
    for user_query, bot_answer in st.session_state.conversation_history:
        st.write(f"**You:** {user_query}")
        st.write(f"**Bot:** {bot_answer}")
        st.write("---")

# Display chat history in a read-only text area
chat_history_text = "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.conversation_history])
st.text_area("Chat History", value=chat_history_text, height=400, disabled=True)
