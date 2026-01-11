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

# ==================== UI ====================
st.markdown("""
# 🧠 MedCopilot V5 — Hybrid Hospital AI  
### Evidence-Based Hospital AI + Global Medical Research  
⚠ Research support only. Not a substitute for professional medical advice.
""")

# ==================== Sidebar ====================
st.sidebar.title("🏥 MedCopilot Status")

PDF_FOLDER = "medical_library"
pdf_files = []

if os.path.exists(PDF_FOLDER):
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if pdf_files:
    st.sidebar.success("Medical Library Loaded")
else:
    st.sidebar.warning("No Medical Library Found")
    st.sidebar.info("Global AI Mode Enabled")

# ==================== Load Models ====================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ==================== Load PDFs ====================
documents = []
sources = []

if pdf_files:
    for file in pdf_files:
        try:
            reader = PdfReader(os.path.join(PDF_FOLDER, file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text) > 200:
                    documents.append(text)
                    sources.append(f"{file} — Page {i+1}")
        except:
            st.warning(f"Skipping corrupted PDF: {file}")

# ==================== Build Vector DB ====================
if documents:
    embeddings = embedder.encode(documents, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
else:
    index = None

# ==================== Workspace ====================
st.markdown("## 🔬 Clinical Research Workspace")

query = st.text_input("Ask a clinical research question:")

ai_mode = st.radio(
    "Select AI Mode:",
    ["🏥 Hospital Evidence AI", "🌍 Global Research AI", "⚡ Hybrid AI"],
    horizontal=True
)

run = st.button("🧠 Run Clinical Intelligence")

# ==================== Hybrid AI Engine ====================
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
