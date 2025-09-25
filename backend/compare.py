# ./backend/compare.py

import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from rich.console import Console
from rich.table import Table

# -------------------------------
# Paths
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTORDB_DIR = os.path.join(ROOT_DIR, "vectordb")

# -------------------------------
# Load Vector DB
# -------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(VECTORDB_DIR):
    raise RuntimeError(f"VectorDB not found at {VECTORDB_DIR}. Please run ingestion.py first.")

vectordb = Chroma(
    persist_directory=VECTORDB_DIR,
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# Fetch car document
# -------------------------------
def fetch_car_doc(make: str, model: str):
    query = f"{make} {model}"
    try:
        results = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return None

    if not results:
        return None

    for doc in results:
        if isinstance(doc, Document):
            content = doc.page_content.lower()
            if make.lower() in content and model.lower() in content:
                return doc
    return results[0] if results else None

# -------------------------------
# Parse Document
# -------------------------------
def parse_doc(doc: Document):
    data = {}
    lines = doc.page_content.splitlines()
    if len(lines) < 2:
        lines = doc.page_content.split(".")
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data

# -------------------------------
# Compare two cars
# -------------------------------
def compare_cars(car1: tuple[str, str], car2: tuple[str, str], return_dataframe: bool = True):
    """
    Compare two cars. Returns a pandas DataFrame (default) for Streamlit,
    or a rich.Table if return_dataframe=False (for CLI use).
    """
    make1, model1 = car1
    make2, model2 = car2

    doc1 = fetch_car_doc(make1, model1)
    doc2 = fetch_car_doc(make2, model2)

    if not doc1 and not doc2:
        return f"No information found for '{make1} {model1}' or '{make2} {model2}'."
    if not doc1:
        return f"No information found for '{make1} {model1}'."
    if not doc2:
        return f"No information found for '{make2} {model2}'."

    data1 = parse_doc(doc1)
    data2 = parse_doc(doc2)

    keys = sorted(set(list(data1.keys()) + list(data2.keys())))

    # Prepare data for DataFrame
    rows = []
    for key in keys:
        val1 = data1.get(key, "-")
        val2 = data2.get(key, "-")

        # Highlight numeric comparisons (optional: can remove formatting for DataFrame)
        try:
            num1 = float(val1.replace(",", "").split()[0])
            num2 = float(val2.replace(",", "").split()[0])
            if num1 > num2:
                val1 = f"{val1} ↑"
                val2 = f"{val2} ↓"
            elif num2 > num1:
                val2 = f"{val2} ↑"
                val1 = f"{val1} ↓"
        except:
            pass

        rows.append([key, val1, val2])

    if return_dataframe:
        df = pd.DataFrame(rows, columns=["Attribute", f"{make1} {model1}", f"{make2} {model2}"])
        return df
    else:
        table = Table(title=f"Car Comparison: {make1} {model1} vs {make2} {model2}")
        table.add_column("Attribute", style="bold")
        table.add_column(f"{make1} {model1}", style="cyan")
        table.add_column(f"{make2} {model2}", style="magenta")
        for row in rows:
            table.add_row(*row)
        return table

# -------------------------------
# CLI Usage
# -------------------------------
if __name__ == "__main__":
    console = Console()
    console.print("[bold yellow]=== Car Comparison ===[/bold yellow]")
    make1 = input("Enter first car Make: ").strip()
    model1 = input("Enter first car Model: ").strip()
    make2 = input("Enter second car Make: ").strip()
    model2 = input("Enter second car Model: ").strip()

    comparison_table = compare_cars((make1, model1), (make2, model2), return_dataframe=False)
    if isinstance(comparison_table, Table):
        console.print(comparison_table)
    else:
        console.print(f"[red]{comparison_table}[/red]")
