from google.colab import drive
drive.mount('/content/drive/')
     

"""
!pip install langchain
!pip install openai
!pip install tiktoken
!pip install faiss-gpu
!pip install langchain_experimental
!pip install "langchain[docarray]"
"""
#!pip install -U langchain-openai
#To use it run `pip install -U langchain-openai` and import as `from langchain_openai import ChatOpenAI`.

     

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import tiktoken
from langchain_openai import ChatOpenAI
print("done importing")
     

import os

# Prompt the user for their OpenAI API key
#api_key = input("Please enter your OpenAI API key: ")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

# Optionally, check that the environment variable was set correctly
print("OPENAI_API_KEY has been set!")

     
OPENAI_API_KEY has been set!

llm_model = "gpt-3.5-turbo"
     

from langchain.text_splitter import CharacterTextSplitter
print("done importing the CharacterTextSplitter")
     
done importing the CharacterTextSplitter

#txt_file_path = '/content/drive/MyDrive/scalexi.txt'
txt_file_path = '/content/drive/MyDrive/SJ-K.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)
     

#Data
print("data")
data
     

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

     

query = "What is the status of Danish Politics?"
result = conversation_chain({"question": query})
answer = result["answer"]
answer
     
"I don't have specific information on the current status of Danish Politics. For the most up-to-date and accurate information, I recommend checking reputable news sources or official government websites related to Denmark."

query = "How can Schultz Jørgensen & Kom help me understand the status of Danish politics?"
result = conversation_chain({"question": query})
answer = result["answer"]
answer
     
'Schultz Jørgensen & Kom can assist you in understanding the status of Danish politics through their strategic consulting services. They have expertise in organizational dynamics, media logic, cultural potentials, and societal political processes. They can provide insights, analysis, and guidance on the current political landscape in Denmark, helping you navigate and comprehend the complexities of Danish politics. Feel free to reach out to them via email at contact@sj-k.dk or by phone at +45 60137260 for more information on how they can specifically tailor their services to your needs regarding Danish politics.'

#!pip install Streamlit
     

from ipywidgets import widgets
from IPython.display import display, clear_output

# Dummy function to simulate conversation chain interaction
# Replace this with your actual function to get responses
def get_response(question):
    # Assuming your conversation chain setup is correctly initialized
    # and named `conversation_chain`
    result = conversation_chain({"question": question})
    answer = result["answer"]
    return answer


# Text input widget
text_input = widgets.Text(
    value='',
    placeholder='Ask me anything:',
    description='Question:',
    disabled=False
)

# Button widget
button_send = widgets.Button(
    description='Submit',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Send',
    icon='check' # FontAwesome icons
)

# Output widget for displaying responses
output = widgets.Output()

def on_button_clicked(b):
    # Display the response when the button is clicked
    with output:
        clear_output()  # Clear the previous output
        print(get_response(text_input.value))

button_send.on_click(on_button_clicked)

# Display the widgets
display(text_input, button_send, output)
