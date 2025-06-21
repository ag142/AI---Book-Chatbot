# # import streamlit as st
# # from dotenv import load_dotenv
# # from PyPDF2 import PdfReader
# # # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.text_splitters import RecursiveCharacterTextSplitter

# # # from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# # # from langchain.vectorstores import FAISS
# # # from langchain.chat_models import ChatOpenAI
# # from langchain.memory import ConversationBufferMemory
# # from langchain.chains import ConversationalRetrievalChain
# # from htmlTemplates import css, bot_template, user_template
# # # from langchain.llms import HuggingFaceHub

# # from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.chat_models import ChatOpenAI
# # from langchain_community.llms import HuggingFaceHub


# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text


# # def get_text_chunks(text):
# #     # text_splitter = CharacterTextSplitter(
# #     #     separator="\n",
# #     #     chunk_size=1000,
# #     #     chunk_overlap=200,
# #     #     length_function=len
# #     # )
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# #     chunks = text_splitter.split_text(text)
# #     return chunks


# # def get_vectorstore(text_chunks):
# #     # embeddings = OpenAIEmbeddings()
# #     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") code chnged old

# #     embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #myself new


# #     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
# #     return vectorstore


# # def get_conversation_chain(vectorstore):
# #     # llm = ChatOpenAI()
# #     #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})#i have commnted this code old

# #     llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})#i myself new


# #     memory = ConversationBufferMemory(
# #         memory_key='chat_history', return_messages=True)
# #     conversation_chain = ConversationalRetrievalChain.from_llm(
# #         llm=llm,
# #         retriever=vectorstore.as_retriever(),
# #         memory=memory
# #     )
# #     return conversation_chain


# # def handle_userinput(user_question):
# #     response = st.session_state.conversation({'question': user_question})
# #     st.session_state.chat_history = response['chat_history']

# #     for i, message in enumerate(st.session_state.chat_history):
# #         if i % 2 == 0:
# #             st.write(user_template.replace(
# #                 "{{MSG}}", message.content), unsafe_allow_html=True)
# #         else:
# #             st.write(bot_template.replace(
# #                 "{{MSG}}", message.content), unsafe_allow_html=True)


# # def main():
# #     load_dotenv()
# #     st.set_page_config(page_title="Chat with multiple PDFs",
# #                        page_icon=":books:")
# #     st.write(css, unsafe_allow_html=True)

# #     if "conversation" not in st.session_state:
# #         st.session_state.conversation = None
# #     if "chat_history" not in st.session_state:
# #         st.session_state.chat_history = None

# #     st.header("Chat with multiple PDFs :books:")
# #     user_question = st.text_input("Ask a question about your documents:")
# #     if user_question:
# #         handle_userinput(user_question)

# #     with st.sidebar:
# #         st.subheader("Your documents")
# #         pdf_docs = st.file_uploader(
# #             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
# #         if st.button("Process"):
# #             with st.spinner("Processing"):
# #                 # get pdf text
# #                 raw_text = get_pdf_text(pdf_docs)

# #                 # get the text chunks
# #                 text_chunks = get_text_chunks(raw_text)

# #                 # create vector store
# #                 vectorstore = get_vectorstore(text_chunks)

# #                 # create conversation chain
# #                 st.session_state.conversation = get_conversation_chain(
# #                     vectorstore)


# # if __name__ == '__main__':
# #     main()

# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.llms import HuggingFaceHub

# from htmlTemplates import css, bot_template, user_template


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = HuggingFaceHub(
#         repo_id="google/flan-t5-base",
#         model_kwargs={"temperature": 0.5, "max_length": 512}
#     )

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever()
#     )
#     return conversation_chain


# # def handle_userinput(user_question):
# #     response = st.session_state.conversation({'question': user_question})
# #     for i, message in enumerate(response['chat_history']):
# #         if i % 2 == 0:
# #             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
# #         else:
# #             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# # def main():
# #     load_dotenv()
# #     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
# #     st.write(css, unsafe_allow_html=True)

# #     if "conversation" not in st.session_state:
# #         st.session_state.conversation = None

# #     st.header("Chat with multiple PDFs :books:")
# #     user_question = st.text_input("Ask a question about your documents:")
# #     if user_question and st.session_state.conversation:
# #         handle_userinput(user_question)

# #     with st.sidebar:
# #         st.subheader("Your documents")
# #         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
# #         if st.button("Process") and pdf_docs:
# #             with st.spinner("Processing..."):
# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 vectorstore = get_vectorstore(text_chunks)
# #                 st.session_state.conversation = get_conversation_chain(vectorstore)
# #                 st.success("PDFs processed. You can now ask questions!")


# # if __name__ == '__main__':
# #     main()

# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# from htmlTemplates import css, bot_template, user_template


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_qa_chain(vectorstore):
#     llm = HuggingFaceHub(
#         repo_id="google/flan-t5-base",
#         model_kwargs={"temperature": 0.5, "max_length": 512}
#     )
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         return_source_documents=True
#     )
#     return qa_chain


# def handle_userinput(user_question):
#     response = st.session_state.qa_chain.run(user_question)
#     st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
#     st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "qa_chain" not in st.session_state:
#         st.session_state.qa_chain = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.qa_chain:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process") and pdf_docs:
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.qa_chain = get_qa_chain(vectorstore)
#                 st.success("PDFs processed. You can now ask questions!")


# if __name__ == '__main__':
#     main()

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Community-based tools
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

from htmlTemplates import css, user_template, bot_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("pdf_index"):
        return FAISS.load_local("pdf_index", embeddings)
    vs = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vs.save_local("pdf_index")
    return vs


def get_most_relevant_chunk(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=1)
    return docs[0].page_content if docs else ""


def ask_llm(context, question):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 256}
    )
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return llm.invoke(prompt)


def handle_userinput(user_question):
    context = get_most_relevant_chunk(user_question, st.session_state.vectorstore)
    answer = ask_llm(context, user_question)
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF ChatBot", page_icon="📄")
    st.write(css, unsafe_allow_html=True)
    st.title("Chat with Your PDFs 🤖")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    user_question = st.text_input("Ask a question based on your uploaded docs:")
    if user_question and st.session_state.vectorstore:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your PDFs…"):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("✅ Done – you can now ask questions!")

if __name__ == "__main__":
    main()
