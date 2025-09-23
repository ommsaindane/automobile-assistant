# ./backend/ingestion.py

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectordb")

def load_pdfs():
    pdfs = [
        "Automotive Engineering Powertrain, Chassis System and Vehicle Body.pdf",
        "Modern Vehicle Design.pdf"
    ]
    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(os.path.join(DATA_DIR, pdf))
        docs.extend(loader.load())
    return docs

def load_csv():
    csv_path = os.path.join(DATA_DIR, "Car Dataset 1945-2020.csv")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    return loader.load()

def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def clear_vectordb():
    if os.path.exists(VECTORDB_DIR):
        shutil.rmtree(VECTORDB_DIR)
        print(f"Cleared existing vectordb at {VECTORDB_DIR}")

def main():
    # Clear existing vectordb before ingestion
    clear_vectordb()

    # Load data
    pdf_docs = load_pdfs()
    csv_docs = load_csv()
    all_docs = pdf_docs + csv_docs

    # Split
    chunks = split_docs(all_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORDB_DIR
    )

    print(f"Ingested {len(chunks)} chunks into ChromaDB at {VECTORDB_DIR}")

if __name__ == "__main__":
    main()
