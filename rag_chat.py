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
st.title("Speciale RAG-model")
st.markdown("""
Velkommen til Jans speciale RAG-model. Det er et værktøj designet til hurtigt at forstå key points fra specialet uden at du skal læse hele specialet. Klik på et af sprøgsmålene nedenfor eller skriv dit eget i tekstfeltet nedenfor.
""")

# Predefined questions
queries = {
    "Hvad er hovedkonklusionerne i specialet?": "Hvad er hovedkonklusionerne i specialet?",
    "Hvilken metode er benyttet?": "Hvilken metode er benyttet?",
    "Beskriv den konceptuelle kontekst for public affairs": "Beskriv den konceptuelle kontekst for public affairs",
    "Hvilken rolle havde Mærsk McKinney Møller Center for Zero Carbon Shipping i specialet?": "Hvilken rolle havde Zero Carbon Shipping i specialet?",
}

st.markdown("""
### Select a Question
Click on a question to get an answer, or type your own question in the input box below.
""")

# Initialize a variable to hold the selected query
selected_query = None

# Create buttons for each predefined query
for button_label, query in queries.items():
    if st.button(button_label):
        selected_query = query
        break  # Stop checking other buttons once one has been pressed

# Check if a query has been selected
if selected_query:
    answer = get_answer(selected_query)
    st.write(f"**Spørgsmål:** {selected_query}")
    st.write("**Svar:**")
    st.write(answer)

# User input
user_query = st.text_input("Stil dit spørgsmål her:")

if user_query:
    answer = get_answer(user_query)
    st.write("**Spørgsmål:**")
    st.write(user_query)
    st.write("**Svar:**")
    st.write(answer)

# Function to clear selection
def clear_selection():
    st.session_state.selected_query = None

st.markdown("____________________")

if st.button("Ryd alt"):
    clear_selection()

# Read the contents of the file
with open(txt_file_path, 'r', encoding='utf-8') as file:
    sjk_text = file.read()

# Dropdown (expander) for displaying non-editable information
with st.expander("Transparency: See the information used in the model here:"):
    st.write(sjk_text)
