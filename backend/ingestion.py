# ./backend/ingestion.py

import os
import shutil
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "..", "vectordb")

# -------------------------------
# CSV to Text Conversion
# -------------------------------
def csv_to_text(csv_path: str):
    df = pd.read_csv(csv_path)
    # select relevant columns
    columns_to_use = ["engine_hp", "capacity_cm3", "transmission",
                      "drive_wheels", "max_speed_km/h"]
    text_rows = []
    for idx, row in df.iterrows():
        parts = []
        for col in columns_to_use:
            val = row.get(col)
            if pd.notna(val) and val != "":
                parts.append(f"{col.replace('_',' ')}: {val}")
        if parts:
            text_rows.append(Document(page_content=". ".join(parts) + ".", metadata={"source": f"csv_row_{idx}"}))
    return text_rows

# -------------------------------
# Load PDFs
# -------------------------------
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

# -------------------------------
# Split Documents
# -------------------------------
def split_docs(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# -------------------------------
# Clear previous Vector DB
# -------------------------------
def clear_vectordb():
    if os.path.exists(VECTORDB_DIR):
        shutil.rmtree(VECTORDB_DIR)
        print(f"Cleared existing vectordb at {VECTORDB_DIR}")

# -------------------------------
# Main Ingestion
# -------------------------------
def main():
    # 1. Clear previous vector DB
    clear_vectordb()

    # 2. Load PDFs
    pdf_docs = load_pdfs()

    # 3. Convert CSV rows to text documents
    csv_docs = csv_to_text(os.path.join(DATA_DIR, "Car Dataset 1945-2020.csv"))

    # 4. Combine all docs
    all_docs = pdf_docs + csv_docs

    # 5. Split into chunks
    chunks = split_docs(all_docs)

    # 6. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 7. Add to Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORDB_DIR
    )

    print(f"Ingested {len(chunks)} chunks into ChromaDB at {VECTORDB_DIR}")

if __name__ == "__main__":
    main()
