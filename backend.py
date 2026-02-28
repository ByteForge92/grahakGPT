import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings


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

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@st.cache_resource
def load_vector_store():
    chunks = load_chunks()
    embedding_model = load_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="consumer_protection_kb"
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


prompt_template = """
You are a helpful consumer rights assistant for India.
Read the context carefully and answer the question in 2-3 clear sentences.
Give a direct, specific answer. Do not copy the context word for word.

Context:
{context}

Question: {question}

Answer in 2-3 sentences:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


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


def get_vector_store():
    return load_vector_store()


def get_answer(query: str) -> str:
    """Answer user queries using the RAG pipeline."""
    rag_chain = load_rag_chain()
    return rag_chain.invoke(query)


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
