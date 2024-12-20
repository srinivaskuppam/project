import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
os.environ["OPENAI_API_KEY"] = "" # Put your key here 

# Load PDF and set up retrieval chain
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
pdf_loader = PyPDFLoader('data/policy-booklet-0923.pdf')
docs = pdf_loader.load()

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit user interface
st.title('PDF Query Chatbot for policy-booklet-0923.pdf')
user_query = st.text_input("Please enter your question:")

# print(user_query)

if user_query:
    response = retrieval_chain.invoke({"input": user_query})
    st.write("Answer:", response["answer"])
    if st.checkbox("Show Context"):
        st.write("Context:", response["context"])
