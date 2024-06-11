import streamlit as st
import fitz  # PyMuPDF
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import os

# Set API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

# Initialize LLM model
llm_model = "gpt-3.5-turbo"

# List of PDF files
pdf_file_paths = [
    'Analyse.pdf', 'Anbefalinger.pdf', 'Implikationer.pdf', 
    'Introduktion.pdf', 'Introduktion_til_AI_redskabet.pdf', 
    'Konceptuel_kombination_af_public_affairs.pdf', 
    'Konceptuel_kontekst_for_GenAI_og_sprog.pdf', 
    'Konceptuel_kontekst_for_public_affairs.pdf', 
    'Konklusion.pdf', 'Metode.pdf', 
    'Præsentation_af_organisationerne.pdf'
]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load and split text from PDFs with metadata
all_documents = []
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for file_path in pdf_file_paths:
    text = extract_text_from_pdf(file_path)
    split_data = text_splitter.split_text(text)
    
    # Add metadata to each split document
    for chunk in split_data:
        all_documents.append({"text": chunk, "metadata": {"source": file_path}})

# Create vector store with domain-specific embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_documents, embedding=embeddings)

# Create conversation chain with improved parameters
llm = ChatOpenAI(temperature=0.5, top_p=0.9, model_name=llm_model)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

def get_answer(query):
    result = conversation_chain({"question": query})
    source = result["source_documents"][0].metadata["source"]
    answer = result["answer"]
    return answer, source

# Streamlit app layout
st.title("Thesis Assistant: SJ-K RAG Model")
st.markdown("""
Welcome to the Thesis Assistant for SJ-K. This tool is designed to help you quickly understand the key points from the thesis without needing to read the entire document. Simply click on a predefined question or enter your own to get started.
""")

# Predefined questions
queries = {
    "What is SJ&K?": "What is SJ&K?",
    "How can they help me?": "How can they help me?",
    "What is the status of Danish politics?*": "What is the status of Danish politics?",
    "How can SJ&K help me understand the status of Danish politics?": "How can SJ&K help me understand the status of Danish politics?",
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
    answer, source = get_answer(selected_query)
    st.write(f"**Query:** {selected_query}")
    st.write("**Answer:**")
    st.write(answer)
    st.write("**Source:**")
    st.write(source)

# User input
user_query = st.text_input("Ask your questions here:")

if user_query:
    answer, source = get_answer(user_query)
    st.write("**Query:**")
    st.write(user_query)
    st.write("**Answer:**")
    st.write(answer)
    st.write("**Source:**")
    st.write(source)

# Function to clear selection
def clear_selection():
    st.session_state.selected_query = None

st.markdown("____________________")

if st.button("Clear"):
    clear_selection()

# Read the contents of the files (for transparency)
pdf_texts = []
for file_path in pdf_file_paths:
    pdf_texts.append(extract_text_from_pdf(file_path))

# Dropdown (expander) for displaying non-editable information
with st.expander("Transparency: See the information used in the model here:"):
    for i, text in enumerate(pdf_texts):
        st.markdown(f"**File {i + 1}: {pdf_file_paths[i]}**")
        st.write(text)

st.markdown("*Note: Information about newsletters, media, or buzz has been excluded.*")
st.markdown("*Asterisk: This question is included to demonstrate that the model does not generate random answers.*")
