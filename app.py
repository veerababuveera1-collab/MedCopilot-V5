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
    page_title="MedCopilot Enterprise — Hospital AI",
    page_icon="🧠",
    layout="wide"
)

# ================== ENTERPRISE UI THEME ==================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f4f9ff, #eaf2ff);
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #64748b;
    font-size: 18px;
    margin-bottom: 25px;
}

.card {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric {
    font-size: 34px;
    font-weight: 700;
    color: #2563eb;
}

.label {
    color: #64748b;
    font-size: 14px;
}

.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}

.upload-box {
    border: 2px dashed #2563eb;
    padding: 25px;
    border-radius: 14px;
    background: rgba(37,99,235,0.05);
    text-align: center;
    font-weight: 600;
    color: #2563eb;
}

.result-box {
    background: rgba(255,255,255,0.95);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.07);
}

</style>
""", unsafe_allow_html=True)

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

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ================== MODEL ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================== HEADER ==================
st.markdown('<div class="main-title">MedCopilot Enterprise</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hospital AI Platform • Clinical Intelligence • Global Medical Research</div>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.title("📁 Medical Knowledge Base")
st.sidebar.markdown('<div class="upload-box">Upload hospital medical PDFs to build AI brain</div>', unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

build_index_btn = st.sidebar.button("🔄 Build Knowledge Index", use_container_width=True)

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
st.sidebar.info(f"📄 Total PDFs in Library: {len(pdf_files)}")

if st.session_state.index_ready:
    st.sidebar.success("🟢 Knowledge Index Ready")
else:
    st.sidebar.warning("🟡 Knowledge Index Not Built")

# ================== PDF UPLOAD ==================
if uploaded_files:
    for f in uploaded_files:
        path = os.path.join(PDF_FOLDER, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully.")

# ================== INDEX BUILDER ==================
def build_index():
    documents = []
    sources = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    progress = st.progress(0)
    total = max(len(pdf_files), 1)

    with st.spinner("🧠 Building hospital knowledge index..."):
        for count, file in enumerate(pdf_files):
            file_path = os.path.join(PDF_FOLDER, file)
            reader = PdfReader(file_path)

            for i, page in enumerate(reader.pages[:200]):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    documents.append(text)
                    sources.append(f"{file} — Page {i+1}")

            progress.progress((count + 1) / total)

    embeddings = embedder.encode(documents, batch_size=16, show_progress_bar=False)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"documents": documents, "sources": sources}, f)

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

if build_index_btn:
    index, docs, srcs = build_index()
    st.session_state.index_ready = True
    st.session_state.documents = docs
    st.session_state.sources = srcs
    st.sidebar.success("✅ Hospital knowledge index built successfully.")

if not st.session_state.index_ready:
    index, docs, srcs = load_index()
    if index is not None:
        st.session_state.index_ready = True
        st.session_state.documents = docs
        st.session_state.sources = srcs

# ================== DASHBOARD METRICS ==================
colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown(f"""
    <div class="card">
        <div class="metric">{len(st.session_state.documents)}</div>
        <div class="label">Hospital Evidence Pages</div>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
    <div class="card">
        <div class="metric">FAISS</div>
        <div class="label">Vector Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

with colC:
    st.markdown("""
    <div class="card">
        <div class="metric">MiniLM</div>
        <div class="label">Medical Embedding Model</div>
    </div>
    """, unsafe_allow_html=True)

with colD:
    st.markdown("""
    <div class="card">
        <div class="metric">LLaMA</div>
        <div class="label">Clinical Reasoning AI</div>
    </div>
    """, unsafe_allow_html=True)

# ================== CLINICAL QUERY ==================
st.markdown("### 🧠 Clinical Intelligence Assistant")

query = st.text_input("Ask a clinical research or decision-support question", placeholder="Eg: Latest ICU sepsis management protocol")

mode = st.radio("AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"], horizontal=True)

run_btn = st.button("🚀 Run Clinical Intelligence", use_container_width=True)

# ================== CLINICAL REASONING ==================
def hospital_clinical_reasoning(query, context):
    prompt = f"""
You are a senior hospital clinical decision support AI.

Using ONLY the hospital evidence below, answer the doctor's question
in a structured medical format with:

- Diagnosis Summary
- Treatment Protocol
- Drug Dosage (if available)
- Monitoring Plan
- Follow-up Plan

Doctor Question:
{query}

Hospital Evidence:
{context}

Rules:
- Use only hospital evidence
- Do not hallucinate
- Be concise and clinical
"""
    result = external_research_answer(prompt)
    return result.get("answer", "No clinical response generated.")

# ================== AI ENGINE ==================
if run_btn and query:
    st.session_state.query_history.append(query)

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if mode == "Hospital AI":
        if not st.session_state.index_ready:
            st.error("❌ Hospital knowledge base not ready. Please upload PDFs and build index.")
        else:
            q_emb = embedder.encode([query])
            index = faiss.read_index(INDEX_FILE)
            D, I = index.search(np.array(q_emb), 5)

            results = [st.session_state.documents[i] for i in I[0]]
            context = "\n\n".join(results)

            with st.spinner("🧠 Generating hospital clinical intelligence..."):
                clinical_answer = hospital_clinical_reasoning(query, context)

            st.subheader("🏥 Hospital Clinical Intelligence")
            st.write(clinical_answer)

            st.download_button("📥 Download Clinical Report", clinical_answer, file_name="clinical_report.txt")

            st.subheader("📚 Evidence Sources")
            for i in I[0]:
                st.info(st.session_state.sources[i])

    elif mode == "Global AI":
        with st.spinner("🌍 Searching global medical research..."):
            ans = external_research_answer(query)

        st.subheader("🌍 Global Medical Research")
        st.write(ans.get("answer", "No response"))

    elif mode == "Hybrid AI":
        output = ""

        if st.session_state.index_ready:
            q_emb = embedder.encode([query])
            index = faiss.read_index(INDEX_FILE)
            D, I = index.search(np.array(q_emb), 3)

            hospital_results = [st.session_state.documents[i] for i in I[0]]
            hospital_context = "\n\n".join(hospital_results)

            hospital_ai = hospital_clinical_reasoning(query, hospital_context)
            output += "### 🏥 Hospital Clinical Intelligence\n\n" + hospital_ai + "\n\n"

        with st.spinner("🌍 Searching global medical research..."):
            ext = external_research_answer(query)

        output += "### 🌍 Global Medical Research\n\n" + ext.get("answer", "No response")

        st.subheader("🧠 Hybrid Clinical Decision Intelligence")
        st.write(output)

        st.download_button("📥 Download Hybrid Report", output, file_name="hybrid_clinical_report.txt")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== QUERY HISTORY ==================
st.sidebar.divider()
st.sidebar.subheader("🕒 Recent Queries")
for q in st.session_state.query_history[-5:]:
    st.sidebar.write("•", q)

# ================== FOOTER ==================
st.divider()
st.caption("MedCopilot Enterprise © Hospital AI Platform | Clinical Decision Intelligence")
