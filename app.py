import os
import re
import tempfile
import streamlit as st
import pdfplumber
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb import PersistentClient
from chromadb.config import Settings

# ========== CONFIG ==========
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"
TOP_K = 3
CHROMA_DIR = "chroma_db"

def check_ollama_available():
    try:
        res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return res.status_code == 200
    except Exception as e:
        st.error("❌ Cannot connect to Ollama server. Make sure the SSH tunnel is active.")
        st.stop()

check_ollama_available()


# ========== FUNCTIONS ==========

def preprocess(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'•', '-', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip() if len(text.strip()) > 40 else None

def chunk_text(text_list, max_tokens=400):
    chunks, current = [], ""
    for text in text_list:
        if len(current.split()) + len(text.split()) <= max_tokens:
            current += " " + text
        else:
            chunks.append(current.strip())
            current = text
    if current:
        chunks.append(current.strip())
    return chunks

def embed_ollama_parallel(texts, model="nomic-embed-text:latest", max_workers=10):
    def embed_one(text):
        try:
            res = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            res.raise_for_status()
            return res.json()["embedding"]
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return [0.0] * 768  # fallback vector
    embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_one, t) for t in texts]
        for future in tqdm(as_completed(futures), total=len(texts), desc="🔗 Embedding"):
            embeddings.append(future.result())
    return embeddings

def stream_llama(prompt):
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True, "temperature": 0.3, "top_p": 0.95},
            stream=True,
            timeout=60
        )
        res.raise_for_status()
        for line in res.iter_lines():
            if line:
                yield line.decode("utf-8").replace("data: ", "")
    except Exception as e:
        st.error(f"Streaming failed: {e}")
        yield ""

# ========== CHROMA INITIALIZATION ==========
os.makedirs(CHROMA_DIR, exist_ok=True)
client = PersistentClient(path=CHROMA_DIR, settings=Settings())
collection = client.get_or_create_collection("mtdp_chunks")

# ========== UI ==========
st.set_page_config("📚 MTDP RAG Assistant", layout="wide")
st.title("🇿🇦 Medium Term Development Plan 2024–2029")
st.markdown("Upload the MTDP PDF, embed it, and ask questions using LLaMA 3.")

# ========== PDF UPLOAD ==========
uploaded = st.file_uploader("📄 Upload PDF", type=["pdf"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded.read())
        pdf_path = tmp_file.name

    st.subheader("🔍 Preprocessing PDF...")
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() for p in pdf.pages]
    texts = [preprocess(t) for t in pages if t]
    clean_texts = list(filter(None, texts))
    chunks = chunk_text(clean_texts)

    st.subheader("💾 Embedding and storing in ChromaDB...")
    embeddings = embed_ollama_parallel(chunks)

    existing_ids = set(collection.get()['ids'])

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"chunk_{i}"
        if chunk_id not in existing_ids:
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": "MTDP 2024-2029", "chunk_index": i}]
            )
    st.success(f"✅ Embedded and stored {len(chunks)} chunks.")

# ========== RAG ==========
st.subheader("🤖 Ask a question about the MTDP:")
query = st.text_input("Enter your question here")
if query:
    with st.spinner("Thinking..."):
        qres = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query}
        )
        qres.raise_for_status()
        qemb = qres.json()["embedding"]

        results = collection.query(
            query_embeddings=[qemb],
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        prompt = (
            f"Answer the question below using only the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        st.markdown("### 📘 Answer")
        placeholder = st.empty()
        full_response = ""
        import json

        for chunk in stream_llama(prompt):
            try:
                data = json.loads(chunk)
                if "response" in data:
                    full_response += data["response"]
                    placeholder.markdown(full_response + "▌")
            except json.JSONDecodeError:
                continue


    with st.expander("🔍 Sources"):
        for i, chunk in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {chunk[:400]}...")
