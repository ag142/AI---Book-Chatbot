# Book Chatbot

## Overview

This project is a web application that enables users to upload PDF documents, extract and summarize their text content, and interact with a Q&A chatbot to get answers related to the document. The application leverages advanced models for text summarization and question-answering to provide valuable insights and answers.

## Features

- **PDF Upload**: Upload PDF files for processing.
- **Text Extraction**: Extract and clean text from the uploaded PDF.
- **Q&A Chatbot**: Ask questions related to the PDF content and receive relevant answers.
- **User-Friendly UI**: Clean and interactive interface with logo integration.

## Technologies

- **Streamlit**: Framework for building the web application.
- **Hugging Face Transformers**: For text summarization and question-answering models.
- **PyPDF2**: For extracting text from PDF files.
- **Python**: Programming language used for development.

## Setup Instructions

### Prerequisites

Ensure Python 3.7 or higher is installed. Check your Python version with:

## USE

 Create a virtual environment (recommended):
````
python -m venv venv
````

 Activate the virtual environment:
````
venv\Scripts\activate
````

4. Install the required packages:
````
pip install -r requirements.txt
````

## Configuration

### 1. Update app.py:

Make sure the path to the logo image and any other local paths are correctly set in the app.py file.

### 2. Set up your models:

Ensure that the models used in QA_chatbot.py are properly downloaded and accessible.

## Usage

### 1. Run the application:
````
streamlit run app.py
````

### 2. Interact with the application:

Upload PDF: Click the "Upload a PDF file" button to upload your PDF document.
Ask Questions: Enter your question in the text input field to get answers based on the PDF content.


### Acknowledgements

Hugging Face: For providing powerful NLP models.
Streamlit: For making it easy to build interactive web applications.

