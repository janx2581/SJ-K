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
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)




def get_answer(query):
    result = conversation_chain({"question": query})
    return result["answer"]


# Streamlit app layout
st.set_page_config(page_title="Thesis Assistant: SJ-K RAG Model", layout="wide")
st.title("Thesis Assistant: SJ-K RAG Model")
st.markdown("""
Welcome to the Thesis Assistant for SJ-K. This tool is designed to help you quickly understand the key points from the thesis without needing to read the entire document. Simply click on a predefined question or enter your own to get started.
""")

# Sidebar for user input
st.sidebar.header("Ask a question")
user_question = st.sidebar.text_area("Enter your question here", height=100)
if st.sidebar.button("Get Answer"):
    if user_question:
        with st.spinner("Generating answer..."):
            answer = get_answer(user_question)
            st.sidebar.write("### Answer")
            st.sidebar.write(answer)
    else:
        st.sidebar.warning("Please enter a question before clicking the button.")

# Main area for chat history
st.subheader("Chat History")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['question']}")
    st.write(f"**Assistant:** {chat['answer']}")

# Function to update chat history
def update_chat_history(question, answer):
    st.session_state.chat_history.append({"question": question, "answer": answer})

# Input box for user to ask questions
st.text_input("Ask another question:", key="input_question", on_change=lambda: update_chat_history(st.session_state.input_question, get_answer(st.session_state.input_question)))
