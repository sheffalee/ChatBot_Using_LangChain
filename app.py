import os
import streamlit as st
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract
import pandas as pd
import openai
import streamlit as st

headers={
    "authorization":st.secrets["API_KEY"],
    "content-type":"application/json"
}

# Set your OpenAI API key
openai.api_key = st.secrets["API_KEY"]



# os.environ["OPENAI_API_KEY"] = "api_key"

# Define Streamlit app title
st.title("Transformers Chatbot")

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
doc = textract.process("Sheffalee_resume_final.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('attention_is_all_you_need.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('attention_is_all_you_need.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a chunk size, adjust as needed.
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])
# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Create conversation chain that uses our vectordb as a retriever, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

# Streamlit UI elements
st.write("Welcome to the Transformers chatbot! Type 'exit' to stop.")
query = st.text_input("Please enter your question:")

if query.lower() == 'exit':
    st.write("Thank you for using the State of the Union chatbot!")
else:
    if st.button("Submit"):
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))

        st.markdown(f'<b>User:</b> {query}', unsafe_allow_html=True)
        st.markdown(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}', unsafe_allow_html=True)


# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import GPT2TokenizerFast
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.chains import ConversationalRetrievalChain

# os.environ["OPENAI_API_KEY"] =

# # Advanced method - Split by chunk

# # Step 1: Convert PDF to text
# import textract
# doc = textract.process("https://github.com/sheffalee/Chatbot_Using_LangChain/blob/main/Academic-Regulations.pdf")

# # Step 2: Save to .txt and reopen (helps prevent issues)
# with open('attention_is_all_you_need.txt', 'w') as f:
#     f.write(doc.decode('utf-8'))

# with open('attention_is_all_you_need.txt', 'r') as f:
#     text = f.read()

# # Step 3: Create function to count tokens
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# def count_tokens(text: str) -> int:
#     return len(tokenizer.encode(text))

# # Step 4: Split text into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 512,
#     chunk_overlap  = 24,
#     length_function = count_tokens,
# )

# chunks = text_splitter.create_documents([text])


# # Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
# type(chunks[0])

# # Get embedding model
# embeddings = OpenAIEmbeddings()

# # Create vector database
# db = FAISS.from_documents(chunks, embeddings)

# from IPython.display import display
# import ipywidgets as widgets

# # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# chat_history = []

# def on_submit(_):
#     query = input_box.value
#     input_box.value = ""

#     if query.lower() == 'exit':
#         print("Thank you for using the State of the Union chatbot!")
#         return

#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))

#     display(widgets.HTML(f'<b>User:</b> {query}'))
#     display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

# print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

# input_box = widgets.Text(placeholder='Please enter your question:')
# input_box.on_submit(on_submit)

# display(input_box)
