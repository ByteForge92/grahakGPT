# app.py
import streamlit as st
from backend import get_answer, get_vector_store

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Consumer Rights Assistant", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Consumer Analytics Assisstant⚖️🏛️")

st.write("Ask questions about consumer protection and get instant answers!")

# Warm up the vector store and models on first load
# (shows a spinner instead of freezing silently)
with st.spinner("⏳ Loading models and knowledge base (first load takes ~1 min)..."):
    vector_store = get_vector_store()

st.write("\n" * 4)

# -----------------------------
# User query input
# -----------------------------
st.write("ENTER YOUR QUESTION HERE:")
st.write("\n" * 2)

query = st.text_input("", "")

if query:
    with st.spinner("🤔 Thinking and generating a response..."):

        # Retrieve top 3 documents
        retrieved_docs = vector_store.similarity_search(query, k=3)

        with st.expander("📄 Retrieved Context"):
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs, start=1):
                    content = doc.page_content
                    keywords = query.lower().split()
                    for kw in keywords:
                        content = content.replace(kw, f"**{kw}**")
                    st.markdown(f"**Doc {i}:**\n\n" + content[:500] + "...\n")

        # Generate answer
        answer = get_answer(query)
        st.subheader("💡 Answer")
        st.write(answer)
