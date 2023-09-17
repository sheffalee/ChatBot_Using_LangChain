import streamlit as st
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import PyPDF2
import openai

# Define Streamlit app title
st.title("Transformers Chatbot")

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["API_KEY"]

# Use the API key in the headers for API requests
headers = {
    "authorization": api_key,
    "content-type": "application/json"
}

# Step 1: Extract text from PDF using PyPDF2 PdfReader
def extract_text_with_pypdf2(pdf_file):
    pdf_text = ''
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    
    return pdf_text

pdf_text = extract_text_with_pypdf2("Sheffalee_resume_final.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
    f.write(pdf_text)

with open('attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
   # Set a chunk size as needed.
   chunk_size=512,
   chunk_overlap=24,
   length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Get embedding model
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Create conversation chain that uses our vectordb as a retriever, this also allows for chat history management

qa = ConversationalRetrievalChain.from_llm(
    llm={
        "llm_type": "transformers",
        "model_name": "gpt2",  # You can use other pretrained models like "gpt2" or "gpt2-medium"
        "api_key": api_key,    # Your OpenAI API key
        "temperature": 0.1,   # Set your desired temperature
        # Add other relevant parameters for the language model if needed
    },
    retriever=db.as_retriever()
)

# qa = ConversationalRetrievalChain.from_llm(
#     llm={
#         "llm_type": "openai",
#         "api_key": api_key,  # Your OpenAI API key
#         "temperature": 0.1,  # Set your desired temperature
#         # Add other relevant parameters for the language model if needed
#     },
#     retriever=db.as_retriever()
# )

# qa = ConversationalRetrievalChain.from_llm(
#     llm={
#         "llm_type": "openai",
#         "api_key": api_key,  # Your OpenAI API key
#         "temperature": 0.1,  # Set your desired temperature
#         # Add other relevant parameters for the language model if needed
#     },
#     retriever=db.as_retriever()
# )
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

# import streamlit as st
# from transformers import GPT2TokenizerFast
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# import PyPDF2
# import openai

# # Define Streamlit app title
# st.title("Transformers Chatbot")

# # Retrieve the API key from Streamlit secrets
# api_key = st.secrets["API_KEY"]

# # Use the API key in the headers for API requests
# headers = {
#     "authorization": api_key,
#     "content-type": "application/json"
# }

# # Step 1: Extract text from PDF using PyPDF2 PdfReader
# def extract_text_with_pypdf2(pdf_file):
#     pdf_text = ''
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
    
#     for page in pdf_reader.pages:
#         pdf_text += page.extract_text()
    
#     return pdf_text

# pdf_text = extract_text_with_pypdf2("Sheffalee_resume_final.pdf")

# # Step 2: Save to .txt and reopen (helps prevent issues)
# with open('attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
#     f.write(pdf_text)

# with open('attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# # Step 3: Create function to count tokens
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# def count_tokens(text: str) -> int:
#     return len(tokenizer.encode(text))

# # Step 4: Split text into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#    # Set a chunk size as needed.
#    chunk_size=512,
#    chunk_overlap=24,
#    length_function=count_tokens,
# )

# chunks = text_splitter.create_documents([text])

# # Get embedding model
# embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# # Create vector database
# db = FAISS.from_documents(chunks, embeddings)

# # Create conversation chain that uses our vectordb as a retriever, this also allows for chat history management
# qa = ConversationalRetrievalChain.from_llm(openai.api_key, db.as_retriever())

# chat_history = []

# # Streamlit UI elements
# st.write("Welcome to the Transformers chatbot! Type 'exit' to stop.")
# query = st.text_input("Please enter your question:")

# if query.lower() == 'exit':
#     st.write("Thank you for using the State of the Union chatbot!")
# else:
#     if st.button("Submit"):
#         result = qa({"question": query, "chat_history": chat_history})
#         chat_history.append((query, result['answer']))

#         st.markdown(f'<b>User:</b> {query}', unsafe_allow_html=True)
#         st.markdown(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}', unsafe_allow_html=True)

# # import streamlit as st
# # from transformers import GPT2TokenizerFast
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain.chains import ConversationalRetrievalChain
# # # import streamlit.secrets as secrets
# # import PyPDF2
# # from pdfminer.high_level import extract_text
# # import openai

# # # Define Streamlit app title
# # st.title("Transformers Chatbot")

# # # Retrieve the API key from Streamlit secrets
# # api_key = st.secrets["API_KEY"]

# # # Use the API key in the headers for API requests
# # headers = {
# #     "authorization": api_key,
# #     "content-type": "application/json"
# # }

# # # Step 1: Extract text from PDF using PyPDF2 PdfReader
# # def extract_text_with_pypdf2(pdf_file):
# #     pdf_text = ''
# #     pdf_reader = PyPDF2.PdfReader(pdf_file)
    
# #     for page in pdf_reader.pages:
# #         pdf_text += page.extract_text()
    
# #     return pdf_text

# # pdf_text = extract_text_with_pypdf2("Sheffalee_resume_final.pdf")

# # # Step 2: Save to .txt and reopen (helps prevent issues)
# # with open('attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
# #     f.write(pdf_text)

# # with open('attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
# #     text = f.read()

# # # Step 3: Create function to count tokens
# # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# # def count_tokens(text: str) -> int:
# #     return len(tokenizer.encode(text))

# # # Step 4: Split text into chunks
# # text_splitter = RecursiveCharacterTextSplitter(
# #    # Set a chunk size as needed.
# #    chunk_size=512,
# #    chunk_overlap=24,
# #    length_function=count_tokens,
# # )

# # chunks = text_splitter.create_documents([text])

# # # Get embedding model
# # # openai.api_key = api_key  # Set the API key for OpenAI

# # embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# # # Create vector database
# # db = FAISS.from_documents(chunks, embeddings)

# # # Create conversation chain that uses our vectordb as a retriever, this also allows for chat history management
# # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1, api_key=openai_api_key), db.as_retriever())

# # chat_history = []

# # # Streamlit UI elements
# # st.write("Welcome to the Transformers chatbot! Type 'exit' to stop.")
# # query = st.text_input("Please enter your question:")

# # if query.lower() == 'exit':
# #     st.write("Thank you for using the State of the Union chatbot!")
# # else:
# #     if st.button("Submit"):
# #         result = qa({"question": query, "chat_history": chat_history})
# #         chat_history.append((query, result['answer']))

# #         st.markdown(f'<b>User:</b> {query}', unsafe_allow_html=True)
# #         st.markdown(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}', unsafe_allow_html=True)

# # # import os
# # # import streamlit as st
# # # from transformers import GPT2TokenizerFast
# # # from langchain.document_loaders import PyPDFLoader
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # from langchain.embeddings import OpenAIEmbeddings
# # # from langchain.vectorstores import FAISS
# # # from langchain.chains.question_answering import load_qa_chain
# # # from langchain.llms import OpenAI
# # # from langchain.chains import ConversationalRetrievalChain
# # # import textract
# # # import pandas as pd
# # # import openai
# # # import streamlit as st
# # # import PyPDF2
# # # from pdfminer.high_level import extract_text
# # # # from chatbot_using_langchain.utils import OpenAIEmbeddings


# # # headers={
# # #     "authorization":st.secrets["API_KEY"],
# # #     "content-type":"application/json"
# # # }

# # # # Set your OpenAI API key
# # # openai.api_key = st.secrets["API_KEY"]

# # # # Define Streamlit app title
# # # st.title("Transformers Chatbot")

# # # # Step 1: Extract text from PDF using PyPDF2 PdfReader
# # # def extract_text_with_pypdf2(pdf_file):
# # #     pdf_text = ''
# # #     pdf_reader = PyPDF2.PdfReader(pdf_file)
    
# # #     for page in pdf_reader.pages:
# # #         pdf_text += page.extract_text()
    
# # #     return pdf_text

# # # pdf_text = extract_text_with_pypdf2("Sheffalee_resume_final.pdf")

# # # # Step 2: Save to .txt and reopen (helps prevent issues)
# # # with open('attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
# # #     f.write(pdf_text)

# # # with open('attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
# # #     text = f.read()

# # # # Step 3: Create function to count tokens
# # # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# # # def count_tokens(text: str) -> int:
# # #     return len(tokenizer.encode(text))

# # # # Step 4: Split text into chunks
# # # text_splitter = RecursiveCharacterTextSplitter(
# # #    # Set a chunk size as needed.
# # #    chunk_size=512,
# # #    chunk_overlap=24,
# # #    length_function=count_tokens,
# # # )

# # # chunks = text_splitter.create_documents([text])


# # # # Get embedding model

# # # openai_api_key = openai.api_key  # Replace with your actual OpenAI API key

# # # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # # # embeddings = OpenAIEmbeddings()

# # # # Create vector database
# # # db = FAISS.from_documents(chunks, embeddings)

# # # # Create conversation chain that uses our vectordb as a retriever, this also allows for chat history management
# # # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# # # chat_history = []

# # # # Streamlit UI elements
# # # st.write("Welcome to the Transformers chatbot! Type 'exit' to stop.")
# # # query = st.text_input("Please enter your question:")

# # # if query.lower() == 'exit':
# # #     st.write("Thank you for using the State of the Union chatbot!")
# # # else:
# # #     if st.button("Submit"):
# # #         result = qa({"question": query, "chat_history": chat_history})
# # #         chat_history.append((query, result['answer']))

# # #         st.markdown(f'<b>User:</b> {query}', unsafe_allow_html=True)
# # #         st.markdown(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}', unsafe_allow_html=True)
