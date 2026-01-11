import streamlit as st
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ==================== CONFIG ====================
st.set_page_config(
    page_title="MedCopilot V5 — Hybrid Hospital AI",
    page_icon="🧠",
    layout="wide"
)

PDF_FOLDER = "medical_library"
os.makedirs(PDF_FOLDER, exist_ok=True)

# ==================== HEADER ====================
st.markdown("""
# 🧠 MedCopilot V5 — Hybrid Hospital AI  
### Evidence-Based Hospital AI + Global Medical Research  
⚠ Research support only. Not a substitute for professional medical advice.
""")

# ==================== SIDEBAR ====================
st.sidebar.title("🏥 MedCopilot Status")

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if pdf_files:
    st.sidebar.success(f"Medical Library Loaded ({len(pdf_files)} PDFs)")
else:
    st.sidebar.warning("No Medical Library Found")
    st.sidebar.info("Global AI Mode Enabled")

# ==================== PDF UPLOAD UI ====================
st.sidebar.markdown("## 📄 Upload Medical PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Drag & Drop Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDFs uploaded successfully. Reloading library...")
    st.experimental_rerun()

# ==================== LOAD EMBEDDING MODEL ====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ==================== LOAD PDFs ====================
documents = []
sources = []

if pdf_files:
    for file in pdf_files:
        file_path = os.path.join(PDF_FOLDER, file)
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 200:
                    documents.append(text)
                    sources.append(f"{file} — Page {i+1}")
        except:
            st.warning(f"⚠ Skipping corrupted PDF: {file}")

# ==================== VECTOR DATABASE ====================
if documents:
    embeddings = embedder.encode(documents, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
else:
    index = None

# ==================== WORKSPACE ====================
st.markdown("## 🔬 Clinical Research Workspace")

query = st.text_input("Ask a clinical research question:")

ai_mode = st.radio(
    "Select AI Mode:",
    ["🏥 Hospital Evidence AI", "🌍 Global Research AI", "⚡ Hybrid AI"],
    horizontal=True
)

run = st.button("🧠 Run Clinical Intelligence")

# ==================== AI ENGINE ====================
if run and query:

    # ---------------- Hospital Evidence Mode ----------------
    if ai_mode == "🏥 Hospital Evidence AI":

        if not documents:
            st.error("No Medical Library Found. Upload PDFs first.")
        else:
            q_embed = embedder.encode([query])
            D, I = index.search(np.array(q_embed), 5)

            context = "\n\n".join([documents[i] for i in I[0]])
            used_sources = [sources[i] for i in I[0]]

            st.markdown("## 🏥 Hospital Evidence-Based Answer")
            st.write(context[:3000])

            st.markdown("### 📚 Evidence Sources")
            for s in used_sources:
                st.info(s)

            st.success("Mode: Hospital Evidence AI")

    # ---------------- Global Research Mode ----------------
    elif ai_mode == "🌍 Global Research AI":

        with st.spinner("🔍 Searching global medical research..."):
            external = external_research_answer(query)

        st.markdown("## 🌍 Global Medical Research Answer")
        st.write(external["answer"])
        st.success("Mode: Global Research AI")

    # ---------------- Hybrid Mode ----------------
    elif ai_mode == "⚡ Hybrid AI":

        response_parts = []

        if documents:
            q_embed = embedder.encode([query])
            D, I = index.search(np.array(q_embed), 3)
            pdf_context = "\n\n".join([documents[i] for i in I[0]])
            response_parts.append("🏥 Hospital Evidence:\n" + pdf_context[:1500])

        external = external_research_answer(query)
        response_parts.append("🌍 Global Research:\n" + external["answer"])

        st.markdown("## ⚡ Hybrid Clinical Intelligence")
        st.write("\n\n".join(response_parts))

        st.success("Mode: Hybrid AI Engine")
