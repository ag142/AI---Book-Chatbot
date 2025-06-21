import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

from htmlTemplates import css, user_template, bot_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)


def get_most_relevant_chunk(user_question, vectorstore):
    docs = vectorstore.similarity_search(user_question, k=1)
    return docs[0].page_content if docs else ""


def get_answer_from_llm(context, question):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return llm.invoke(prompt)


def handle_userinput(user_question):
    context = get_most_relevant_chunk(user_question, st.session_state.vectorstore)
    answer = get_answer_from_llm(context, user_question)
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.write(css, unsafe_allow_html=True)

    st.title("Chat with PDF 📄")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.vectorstore:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Reading & indexing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Done! You can now ask questions.")


if __name__ == "__main__":
    main()
