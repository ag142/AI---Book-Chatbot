import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Extract raw text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# 2. Split text into chunks
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 3. Create FAISS vector store with embeddings
def build_vectorstore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

# 4. Search top matching paragraph
def search_best_match(query, vectorstore):
    result = vectorstore.similarity_search(query, k=1)
    return result[0].page_content if result else "No relevant match found."

# 5. Streamlit app
def main():
    st.set_page_config("Chat with Your PDFs 🤖", page_icon="📄")
    st.title("Chat with Your PDFs 🤖")
    st.write("Ask a question based on your uploaded docs:")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Question input
    query = st.text_input("What would you like to ask?")
    if query and st.session_state.vectorstore:
        with st.spinner("Searching..."):
            response = search_best_match(query, st.session_state.vectorstore)
            st.markdown("### 📌 Best Matching Paragraph:")
            st.write(response)

    # Sidebar PDF uploader
    with st.sidebar:
        st.subheader("📁 Upload PDFs")
        pdf_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
        if st.button("📚 Process"):
            with st.spinner("Reading & indexing..."):
                text = extract_text_from_pdfs(pdf_files)
                chunks = split_into_chunks(text)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.success("Ready to answer your questions!")

if __name__ == "__main__":
    main()


################################## Its working with answering pdf in paragraph ######################################################################