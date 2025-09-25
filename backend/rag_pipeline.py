# ./backend/rag_pipeline.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------------
# Step 0: Setup device
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------------------------------
# Step 1: Load LLaMA 3.2 3B (quantized)
# -------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3b"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

device_map = "auto"  # auto GPU/CPU mapping

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        quantization_config=quant_config
    )

    print("Creating HuggingFace pipeline...")
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
except Exception as e:
    raise RuntimeError(f"Failed to load LLaMA model: {e}")

# -------------------------------
# Step 2: Load Chroma Vector DB
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIR = os.path.join(ROOT_DIR, "vectordb")

print("Loading embeddings and vector database...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
except Exception as e:
    raise RuntimeError(f"Failed to load Chroma vector DB: {e}")

# -------------------------------
# Step 3: Custom Prompt Templates
# -------------------------------
QUESTION_PROMPT = """
You are an expert automobile engineer. Use the provided document context to answer the question.
- Ignore empty fields or missing values.
- Provide concise, clear, and human-readable explanations.
- Do not just dump raw data; explain it.
- If unsure, say "I don't know".

Context:
{context}

Question: {question}
Answer:
"""

COMBINE_PROMPT = """
You are an expert automobile engineer. Several summaries of documents are provided below.
Combine them into a single, clear, concise answer to the question.
- Do not repeat yourself.
- Ignore irrelevant details.
- If unsure, say "I don't know".

Summaries:
{summaries}

Question: {question}
Final Answer:
"""

question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=QUESTION_PROMPT
)

combine_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template=COMBINE_PROMPT
)

# -------------------------------
# Step 4: Setup RetrievalQA with map_reduce
# -------------------------------
def get_qa_chain(top_k: int = 5) -> RetrievalQA:
    """
    Returns a RetrievalQA chain with a retriever that fetches `top_k` documents.
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
        return_source_documents=True
    )

# Default QA chain with k=5
qa_chain = get_qa_chain(top_k=5)

# -------------------------------
# Step 5: Query function
# -------------------------------
def ask_question(question: str, top_k: int = 5) -> tuple[str, list]:
    """
    Ask a question using the RAG pipeline.
    Returns a tuple: (clean_answer, list of source documents)
    """
    try:
        chain = get_qa_chain(top_k=top_k)
        output = chain.invoke({"query": question})

        # Attempt to extract only the final human-readable answer
        answer = output.get("result")
        if isinstance(answer, str):
            # Remove any prompt instructions accidentally included
            # Look for 'Final Answer:' marker if present
            if "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()
        sources = output.get("source_documents", [])
        return answer, sources
    except Exception as e:
        return f"Error during question answering: {e}", []
