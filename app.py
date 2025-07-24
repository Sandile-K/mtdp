import os
import streamlit as st
import requests
import pdfplumber
from tqdm import tqdm
from uuid import uuid4
import chromadb

# Load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = "mistralai/mistral-7b-instruct:free"
CHAT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# Initialize ChromaDB using latest syntax
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("rag_collection")

# Headers for OpenRouter
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://mtdp.onrender.com",
    "X-Title": "MTDP-RAG-App"
}

# Text splitting helper
def chunk_text(text, max_tokens=300):
    chunks = []
    current = ""
    for para in text.split("\n"):
        if len(current) + len(para) < max_tokens:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

# Embedding function via OpenRouter
def get_embedding(text):
    res = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers=headers,
        json={"model": EMBED_MODEL, "input": text}
    )
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]

# LLM chat completion via OpenRouter
def ask_model(query, context):
    prompt = f"""
You are an expert assistant. Based on the following context, answer the user query clearly and accurately.

Context:
{context}

Query:
{query}
"""
    res = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for South Africa's Medium Term Development Plan 2024–2029."},
                {"role": "user", "content": prompt}
            ]
        }
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.set_page_config(page_title="MTDP RAG App")
st.title("📘 MTDP 2024–2029 RAG App (OpenRouter Edition)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        all_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    st.success("PDF loaded.")

    with st.spinner("Embedding and storing chunks..."):
        chunks = chunk_text(all_text)
        for chunk in tqdm(chunks):
            embedding = get_embedding(chunk)
            uid = str(uuid4())
            collection.add(documents=[chunk], ids=[uid], embeddings=[embedding])
        chroma_client.persist()
        st.success(f"{len(chunks)} chunks embedded and stored!")

query = st.text_input("Ask a question about the MTDP document:")

if query:
    with st.spinner("Searching..."):
        query_embedding = get_embedding(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=4)
        context = "\n\n".join(results["documents"][0])
        response = ask_model(query, context)
        st.markdown("### 📎 Answer")
        st.markdown(response)
