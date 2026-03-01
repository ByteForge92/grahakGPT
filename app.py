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
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# Warm-up vector store
# -----------------------------
@st.cache_resource(show_spinner=False)
def warm_up():
    return get_vector_store()

with st.spinner("⏳ Initializing legal knowledge base..."):
    vector_store = warm_up()

# -----------------------------
# Simple Case Type Detection (Feature 5)
# -----------------------------
def detect_case_type(query):
    q = query.lower()

    if "defect" in q or "damaged" in q or "faulty" in q:
        return "Defective Goods"
    elif "refund" in q or "return" in q:
        return "Refund / Replacement Issue"
    elif "service" in q or "delay" in q:
        return "Deficiency in Service"
    elif "compensation" in q or "harassment" in q:
        return "Compensation Claim"
    elif "jurisdiction" in q or "commission" in q:
        return "Court Jurisdiction Query"
    else:
        return "General Consumer Dispute"

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Ask your consumer law question...")

if user_input:

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Analyzing legal provisions and generating response..."):

            # Detect case type
            case_type = detect_case_type(user_input)

            # Retrieve context
            retrieved_docs = vector_store.similarity_search(user_input, k=3)

            # Sidebar context
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

            # Structured Answer Section (Feature 3)
            st.markdown(f"**🗂 Case Type Detected:** {case_type}")
            st.markdown("### 💡 Legal Guidance")
            st.markdown(answer)

            # Copy Answer Button
            st.download_button(
                label="📋 Copy Answer as Text",
                data=answer,
                file_name="consumer_legal_answer.txt",
                mime="text/plain"
            )

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "⚠️ This tool provides general informational assistance based on the Consumer Protection Act, 2019. "
    "It is not a substitute for professional legal advice."
)
