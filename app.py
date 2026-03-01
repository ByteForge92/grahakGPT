import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from backend import get_answer, get_vector_store

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Consumer Court AI Assistant",
    layout="wide"
)

# Subtle styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title Section
# -----------------------------
st.title("Consumer Court AI Assistant ⚖️🏛️")
st.caption("Ask questions about consumer protection law in India and get instant guidance.")

st.divider()

# -----------------------------
# Warm-up vector store (first load)
# -----------------------------
@st.cache_resource(show_spinner=False)
def warm_up():
    return get_vector_store()

with st.spinner("⏳ Initializing legal knowledge base..."):
    vector_store = warm_up()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Ask your consumer law question...")

if user_input:

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Analyzing legal provisions and generating response..."):

            # Retrieve context
            retrieved_docs = vector_store.similarity_search(user_input, k=3)

            # Show retrieved context in sidebar
            with st.sidebar:
                st.subheader("📄 Retrieved Legal Context")
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Doc {i}:**")
                        st.write(doc.page_content[:400] + "...")
                else:
                    st.write("No relevant documents found.")

            # Generate answer
            answer = get_answer(user_input)
            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# Footer Disclaimer
# -----------------------------
st.divider()
st.caption(
    "⚠️ This tool provides general informational assistance based on the Consumer Protection Act, 2019. "
    "It is not a substitute for professional legal advice."
)
