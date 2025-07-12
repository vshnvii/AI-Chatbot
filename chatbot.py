import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

os.environ["GROQ_API_KEY"]="get your own key"

st.header("My Chatbot")

with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload your pdf", type="pdf")

if file is not None:
    pdf_pages=PdfReader(file)
    text="" 
    for page in pdf_pages.pages:
        text+=page.extract_text()
        # st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len
    )

    chunks=text_splitter.split_text(text)
    # st.write(chunks)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embaddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    
    vector_store = FAISS.from_texts(chunks, embaddings)

    user_query=st.text_input("Enter your query")

    if user_query:
        match = vector_store.similarity_search(user_query)
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.0,
            max_retries=2
        )
 
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)
        
        st.header("Answer")
        st.write(response)

    if file.size > 5 * 1024 * 1024:
        st.warning("Please upload a PDF smaller than 5MB.")
        st.stop()
