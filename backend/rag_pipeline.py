# rag_pipeline.py

import warnings
warnings.filterwarnings("ignore")

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# -------------------------------
# Step 1: Load Llama 3.2 3B (quantized)
# -------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config
)

hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------------------------------
# Step 2: Load Chroma Vector DB (root folder)
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIR = os.path.join(ROOT_DIR, "chroma_db")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# Step 3: Set up RAG (RetrievalQA)
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# -------------------------------
# Step 4: Query function
# -------------------------------
def ask_question(question: str):
    return qa_chain.run(question)
