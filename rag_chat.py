import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
import time

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
    try:
        result = conversation_chain({"question": query})
        return result["answer"]
    except Exception as e:
        return f"An error occurred: {e}"

def stream_answer(query):
    answer = get_answer(query)
    for chunk in answer.split(' '):  # Yield word by word
        yield chunk + ' '
        time.sleep(0.01)  # Optional: adjust the sleep time for a smoother typewriter effect

# Custom CSS for dark blue and black color scheme
st.markdown(
    """
    <style>
    body {
        background-color: #000000;  /* Black background */
        color: #ffffff;  /* White text */
    }
    .stApp {
        background-color: #000000;  /* Black background */
    }
    .stMarkdown, .stTextInput, .stButton, .stTitle {
        color: #ffffff;  /* White text */
    }
    .stButton button {
        background-color: #00008B;  /* Dark blue button */
        color: #ffffff;  /* White text on button */
    }
    .stTextInput>div>input {
        background-color: #00008B;  /* Dark blue input box */
        color: #ffffff;  /* White text in input box */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
        color: #ffffff;  /* White headings and paragraphs */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("Speciale RAG-model")
st.markdown("""
Velkommen til Specialeassistenten for Jans speciale. Dette værktøj er designet til at hjælpe dig med hurtigt at forstå de centrale punkter fra specialet uden at skulle læse hele dokumentet. Klik blot på et foruddefineret spørgsmål eller indtast dit eget for at komme i gang.
""")

# Predefined questions
queries = {
    "Beskriv den konceptuelle kontekst for GenAI i specialet. Vær detaljeret": "Beskriv den konceptuelle kontekst for GenAI i specialet. Vær detaljeret",
    "Hvad er konklusionen?": "Hvad er konklusionen?",
    "Opsummer pointerne om public affairs. Vær detaljeret": "Opsummer pointerne om public affairs. Vær detaljeret",
    "Hvilke rolle spillede Mærsk McKinney Møller Center for Zero Carbon Shipping?": "Hvilke rolle spillede Mærsk McKinney Møller Center for Zero Carbon Shipping?",
    "Beskriv metoden benyttet i specialet. Vær detaljeret": "Beskriv metoden benyttet i specialet. Vær detaljeret",
    "Hvordan virker AI-redskabet?": "Hvordan virker AI-redskabet?",
}

st.markdown("""
### Vælg et sprøgsmål eller stil dit eget i tekstboksen nedenfor
""")

# Initialize a variable to hold the selected query
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None

# Create buttons for each predefined query and handle selection
for button_label, query in queries.items():
    if st.button(button_label):
        st.session_state.selected_query = query

# User input
user_query = st.text_input("Stil dit spørgsmål her:")

if user_query:
    st.session_state.selected_query = user_query

# Check if a query has been selected
if st.session_state.selected_query:
    st.write(f"**Spørgsmål:** {st.session_state.selected_query}")
    st.write("**Svar:**")
    st.write_stream(stream_answer(st.session_state.selected_query))

# Function to clear selection
def clear_selection():
    st.session_state.selected_query = None
    st.experimental_rerun()

st.markdown("____________________")

if st.button("Clear"):
    clear_selection()

