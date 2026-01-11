import streamlit as st
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ================= CONFIG =================
st.set_page_config("MedCopilot Enterprise", "🧠", layout="wide")

PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

# ================= UI =================
st.title("🧠 MedCopilot Enterprise — Hospital AI Platform")
st.caption("Evidence-Based Hospital AI + Global Medical Research")

# ================= Load Embedder =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================= Native Fast Text Splitter =================
def split_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ================= Upload PDFs =================
st.sidebar.header("📄 Medical Library")

uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Uploading medical PDFs..."):
        for f in uploaded_files:
            path = os.path.join(PDF_FOLDER, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())

    st.sidebar.success("PDFs uploaded successfully!")
    st.cache_resource.clear()
    st.rerun()

# ================= Index Builder =================
def build_index():
    documents = []
    sources = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    with st.spinner("🧠 Building Medical Knowledge Index..."):
        for file in pdf_files:
            reader = PdfReader(os.path.join(PDF_FOLDER, file))

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue

                chunks = split_text(text)

                for chunk in chunks:
                    if len(chunk) > 200:
                        documents.append(chunk)
                        sources.append(f"{file} — Page {i+1}")

        embeddings = embedder.encode(
            documents,
            batch_size=32,
            show_progress_bar=True
        )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "documents": documents,
            "sources": sources
        }, f)

    return index, documents, sources

# ================= Load Cached Index =================
@st.cache_resource
def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
        index = faiss.read_index(INDEX_FILE)

        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)

        return index, data["documents"], data["sources"]

    elif os.listdir(PDF_FOLDER):
        return build_index()
    else:
        return None, [], []

index, documents, sources = load_index()

# ================= Workspace =================
st.subheader("🔬 Clinical Research Workspace")

query = st.text_input("Ask a clinical research question:")
mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

run = st.button("Run Clinical Intelligence")

# ================= AI Engine =================
if run and query:

    if mode == "Hospital AI":
        if not index:
            st.error("No medical library uploaded.")
        else:
            q_emb = embedder.encode([query])
            D, I = index.search(np.array(q_emb), 5)

            results = []
            for i in I[0]:
                results.append(documents[i])

            context = "\n\n".join(results)

            st.subheader("🏥 Hospital Evidence")
            st.write(context[:3500])

    elif mode == "Global AI":
        with st.spinner("🌍 Searching global medical research..."):
            ans = external_research_answer(query)

        st.subheader("🌍 Global Research")
        st.write(ans["answer"])

    elif mode == "Hybrid AI":
        result = ""

        if index:
            q_emb = embedder.encode([query])
            D, I = index.search(np.array(q_emb), 3)

            hospital_results = []
            for i in I[0]:
                hospital_results.append(documents[i])

            hospital_context = "\n\n".join(hospital_results)
            result += "🏥 Hospital Evidence:\n\n" + hospital_context[:1800] + "\n\n"

        with st.spinner("🌍 Searching global medical research..."):
            ext = external_research_answer(query)

        result += "🌍 Global Research:\n\n" + ext["answer"]

        st.subheader("🧠 Hybrid Clinical Intelligence")
        st.write(result)
