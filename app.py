import streamlit as st
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MedCopilot Enterprise",
    page_icon="🧠",
    layout="wide"
)

# ================== STORAGE ==================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ================== SESSION ==================
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "documents" not in st.session_state:
    st.session_state.documents = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# ================== MODEL ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================== UI HEADER ==================
st.title("🧠 MedCopilot Enterprise — Hospital AI Platform")
st.caption("Clinical Evidence • Medical Intelligence • Global Research")

# ================== SIDEBAR ==================
st.sidebar.header("📁 Medical Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

build_index_btn = st.sidebar.button("🔄 Build Knowledge Index")

# ================== PDF UPLOAD ==================
if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(PDF_FOLDER, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

    st.sidebar.success("PDFs uploaded successfully.")

# ================== INDEX BUILDER ==================
def build_index():
    documents = []
    sources = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    with st.spinner("🧠 Building hospital knowledge index..."):
        for file in pdf_files:
            reader = PdfReader(os.path.join(PDF_FOLDER, file))

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 300:
                    documents.append(text)
                    sources.append(f"{file} — Page {i+1}")

        embeddings = embedder.encode(
            documents,
            batch_size=32,
            show_progress_bar=False
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

# ================== LOAD INDEX ==================
@st.cache_resource
def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return index, data["documents"], data["sources"]
    return None, [], []

# ================== BUILD INDEX BUTTON ==================
if build_index_btn:
    index, docs, srcs = build_index()
    st.session_state.index_ready = True
    st.session_state.documents = docs
    st.session_state.sources = srcs
    st.sidebar.success("✅ Hospital knowledge index built successfully.")

# ================== LOAD EXISTING INDEX ==================
if not st.session_state.index_ready:
    index, docs, srcs = load_index()
    if index is not None:
        st.session_state.index_ready = True
        st.session_state.documents = docs
        st.session_state.sources = srcs

# ================== MAIN DASHBOARD ==================
st.divider()
st.subheader("🔬 Clinical Intelligence Dashboard")

col1, col2 = st.columns([3, 1])

with col2:
    mode = st.radio(
        "AI Mode",
        ["Hospital AI", "Global AI", "Hybrid AI"]
    )

with col1:
    query = st.text_input("Ask a clinical research question")

run_btn = st.button("🚀 Run Clinical Intelligence")

# ================== RESULT PANEL ==================
result_panel = st.container()

# ================== AI ENGINE ==================
if run_btn and query:

    with result_panel:
        st.subheader("📊 Clinical Intelligence Result")

        # ---------------- Hospital AI ----------------
        if mode == "Hospital AI":
            if not st.session_state.index_ready:
                st.error("❌ Hospital knowledge base not ready. Please upload PDFs and build index.")
            else:
                q_emb = embedder.encode([query])
                D, I = faiss.read_index(INDEX_FILE).search(np.array(q_emb), 5)

                results = []
                for i in I[0]:
                    results.append(st.session_state.documents[i])

                context = "\n\n".join(results)

                st.markdown("### 🏥 Hospital Evidence")
                st.write(context[:3500])

        # ---------------- Global AI ----------------
        elif mode == "Global AI":
            with st.spinner("🌍 Searching global medical research..."):
                ans = external_research_answer(query)

            st.markdown("### 🌍 Global Medical Research")
            st.write(ans.get("answer", "No response"))

        # ---------------- Hybrid AI ----------------
        elif mode == "Hybrid AI":
            output = ""

            if st.session_state.index_ready:
                q_emb = embedder.encode([query])
                D, I = faiss.read_index(INDEX_FILE).search(np.array(q_emb), 3)

                hospital_results = []
                for i in I[0]:
                    hospital_results.append(st.session_state.documents[i])

                hospital_context = "\n\n".join(hospital_results)
                output += "### 🏥 Hospital Evidence\n\n" + hospital_context[:1800] + "\n\n"

            with st.spinner("🌍 Searching global medical research..."):
                ext = external_research_answer(query)

            output += "### 🌍 Global Medical Research\n\n" + ext.get("answer", "No response")

            st.markdown("### 🧠 Hybrid Clinical Intelligence")
            st.write(output)

# ================== FOOTER ==================
st.divider()
st.caption("MedCopilot Enterprise © Hospital AI Platform | Clinical Decision Intelligence")
