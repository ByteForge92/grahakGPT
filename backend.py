# backend.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline


# -----------------------------
# Load & chunk knowledge base
# (cached so it only runs once)
# -----------------------------
@st.cache_resource
def load_chunks():
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# -----------------------------
# Embeddings
# (cached so model loads once)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


# -----------------------------
# Vector Store
# Built in-memory (Streamlit Cloud
# has an ephemeral filesystem, so
# persisting to disk is unreliable)
# -----------------------------
@st.cache_resource
def load_vector_store():
    chunks = load_chunks()
    embedding_model = load_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="consumer_protection_kb"
        # No persist_directory — stays in memory, cached by Streamlit
    )


# -----------------------------
# LLM
# device=-1 forces CPU (required
# on Streamlit Cloud; no MPS/CUDA)
# -----------------------------
@st.cache_resource
def load_llm():
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,           # CPU — works on Streamlit Cloud
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are a helpful assistant that answers questions based on the provided context.
Only use the information from the context. If the answer is not contained in the context, say you cannot answer.

Context:
{context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# -----------------------------
# Build RAG chain
# -----------------------------
@st.cache_resource
def load_rag_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# -----------------------------
# Public helpers used by app.py
# -----------------------------
def get_vector_store():
    return load_vector_store()


def get_answer(query: str) -> str:
    """Answer user queries using the RAG pipeline."""
    rag_chain = load_rag_chain()
    return rag_chain.invoke(query)


# -----------------------------
# Optional: test via CLI
# -----------------------------
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is a consumer dispute?"
    print(f"\n🔎 Query: {query}\n")
    vs = get_vector_store()
    retrieved_docs = vs.similarity_search(query, k=3)
    print("📄 Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Doc {i} ---\n{doc.page_content[:500]}...\n")
    answer = get_answer(query)
    print("💡 Answer:\n")
    print(answer)
