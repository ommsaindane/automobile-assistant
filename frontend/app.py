# ./frontend/app.py

import requests
import streamlit as st
import pandas as pd
import sys
import os

# -------------------------------
# Add project root to sys.path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.compare import compare_cars

# -------------------------------
# Config
# -------------------------------
API_URL = "http://127.0.0.1:8000"

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="Automobile Knowledge Assistant", page_icon="🚗")
st.title("Automobile Knowledge Assistant")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Ask a Question", "Compare Cars", "Health Check"])

# -------------------------------
# Ask a Question Page
# -------------------------------
if page == "Ask a Question":
    st.subheader("Ask about Automobiles")
    query = st.text_area("Enter your question:", height=100)
    top_k = st.slider("Number of documents to retrieve (top_k)", min_value=1, max_value=10, value=3)

    if st.button("Ask"):
        if query.strip():
            with st.spinner("Fetching answer from backend..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"query": query, "top_k": top_k},
                        timeout=60
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["answer"])
                        if data.get("sources"):
                            st.info(f"Sources: {', '.join(data['sources'])}")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

# -------------------------------
# Compare Cars Page
# -------------------------------
elif page == "Compare Cars":
    st.subheader("Compare Two Cars")

    col1, col2 = st.columns(2)
    with col1:
        make1 = st.text_input("Car 1 Make")
        model1 = st.text_input("Car 1 Model")
    with col2:
        make2 = st.text_input("Car 2 Make")
        model2 = st.text_input("Car 2 Model")

    if st.button("Compare"):
        if make1 and model1 and make2 and model2:
            with st.spinner("Comparing cars..."):
                try:
                    comparison_df = compare_cars((make1, model1), (make2, model2))
                    if isinstance(comparison_df, pd.DataFrame):
                        # Highlight numeric advantages
                        def highlight_max(row):
                            mask = []
                            numeric_vals = []
                            for col in row.index[1:]:
                                try:
                                    val = float(str(row[col]).split()[0].replace(",", ""))
                                    numeric_vals.append(val)
                                except:
                                    numeric_vals.append(float("-inf"))
                            max_val = max(numeric_vals)
                            for idx, col in enumerate(row.index[1:]):
                                mask.append("background-color: lightgreen" if numeric_vals[idx] == max_val else "")
                            return pd.Series([""] + mask, index=row.index)

                        st.dataframe(comparison_df.style.apply(highlight_max, axis=1))
                    else:
                        st.text(comparison_df)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
        else:
            st.warning("Please enter Make and Model for both cars.")

# -------------------------------
# Health Check Page
# -------------------------------
elif page == "Health Check":
    st.subheader("Backend Health Check")
    try:
        res = requests.get(f"{API_URL}/health", timeout=10)
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error(f"Backend error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Failed to connect: {e}")
