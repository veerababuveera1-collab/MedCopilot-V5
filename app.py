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
SOURCES_FILE = os.path.join(VECTOR_FOLDER, "sources.pkl")

# ================= UI =================
st.title("🧠 MedCopilot Enterprise — Hospital AI Platform")
st.caption("Evidence-Based Hospital AI + Global Medical Research")

# ================= Load Model Once =================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================= Upload PDFs =================
st.sidebar.header("📄 Medical Library")

uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(PDF_FOLDER, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success("PDFs uploaded. Building index...")
    st.rerun()

# ================= Index Builder =================
def build_index():
    documents = []
    sources = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(PDF_FOLDER, file))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text) > 200:
                        documents.append(text)
                        sources.append(f"{file} — Page {i+1}")
            except:
                pass

    embeddings = embedder.encode(documents, show_progress_bar=False)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)
    with open(SOURCES_FILE, "wb") as f:
        pickle.dump(sources, f)

    return index, documents, sources

# ================= Load Cached Index =================
if os.path.exists(INDEX_FILE) and os.path.exists(SOURCES_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(SOURCES_FILE, "rb") as f:
        sources = pickle.load(f)

    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(PDF_FOLDER, file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text and len(text) > 200:
                        documents.append(text)
            except:
                pass
else:
    if os.listdir(PDF_FOLDER):
        index, documents, sources = build_index()
    else:
        index = None
        documents = []
        sources = []

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

            context = "\n\n".join([documents[i] for i in I[0]])
            st.write(context[:3000])

    elif mode == "Global AI":
        ans = external_research_answer(query)
        st.write(ans["answer"])

    elif mode == "Hybrid AI":
        result = ""

        if index:
            q_emb = embedder.encode([query])
            D, I = index.search(np.array(q_emb), 3)
            context = "\n\n".join([documents[i] for i in I[0]])
            result += "🏥 Hospital Evidence:\n" + context[:1500] + "\n\n"

        ext = external_research_answer(query)
        result += "🌍 Global Research:\n" + ext["answer"]

        st.write(result)
