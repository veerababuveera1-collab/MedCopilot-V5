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

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ================== MODEL ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ================== UI HEADER ==================
st.title("🧠 MedCopilot Enterprise — Hospital AI Platform")
st.caption("Clinical Evidence • Medical Intelligence • Global Research")

# ================== SIDEBAR ==================
st.sidebar.title("📁 Medical Knowledge Base")
st.sidebar.markdown("Upload hospital medical PDFs and build AI knowledge index.")

uploaded_files = st.sidebar.file_uploader(
    "Upload Medical PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.divider()

build_index_btn = st.sidebar.button(
    "🔄 Build Knowledge Index",
    use_container_width=True
)

# ---- PDF Count ----
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
st.sidebar.divider()
st.sidebar.info(f"📄 Total PDFs in Library: {len(pdf_files)}")

# ---- Index Status ----
if st.session_state.index_ready:
    st.sidebar.success("🟢 Knowledge Index Ready")
else:
    st.sidebar.warning("🟡 Knowledge Index Not Built")

# ---- Auto Rebuild Warning ----
if uploaded_files and st.session_state.index_ready:
    st.sidebar.warning("⚠️ New PDFs uploaded. Please rebuild knowledge index.")

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
    failed_files = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    progress = st.progress(0)
    total = max(len(pdf_files), 1)
    count = 0

    with st.spinner("🧠 Building hospital knowledge index..."):
        for file in pdf_files:
            file_path = os.path.join(PDF_FOLDER, file)

            try:
                reader = PdfReader(file_path)

                for i, page in enumerate(reader.pages):
                    if i > 200:
                        break

                    try:
                        text = page.extract_text()
                        if not text or len(text.strip()) < 100:
                            continue

                        documents.append(text)
                        sources.append(f"{file} — Page {i+1}")
                    except:
                        continue

            except:
                failed_files.append(file)
                continue

            count += 1
            progress.progress(count / total)

        if not documents:
            st.error("❌ No valid text could be extracted from uploaded PDFs.")
            return None, [], []

        embeddings = embedder.encode(
            documents,
            batch_size=16,
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

    if failed_files:
        st.warning(f"⚠️ Skipped corrupted PDFs: {', '.join(failed_files)}")

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
    if index is not None:
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

# ================== MAIN DASHBOARD ==================
st.divider()
st.subheader("🔬 Clinical Intelligence Dashboard")

st.markdown("### 💡 Example Clinical Questions")
st.write("- What are the causes of hypertension?")
st.write("- Latest treatment protocol for Type 2 Diabetes")
st.write("- ICU sepsis management guidelines")

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

    st.session_state.query_history.append(query)

    with result_panel:
        st.subheader("📊 Clinical Intelligence Result")

        # ---------------- Hospital AI ----------------
        if mode == "Hospital AI":
            if not st.session_state.index_ready:
                st.error("❌ Hospital knowledge base not ready. Please upload PDFs and build index.")
            else:
                q_emb = embedder.encode([query])
                D, I = faiss.read_index(INDEX_FILE).search(np.array(q_emb), 5)

                results = [st.session_state.documents[i] for i in I[0]]
                context = "\n\n".join(results)

                with st.spinner("🧠 Generating clinical intelligence..."):
                    clinical_answer = hospital_clinical_reasoning(query, context)

                st.markdown("### 🧠 Hospital Clinical Intelligence")
                st.write(clinical_answer)

                st.download_button(
                    "📥 Download Clinical Report",
                    clinical_answer,
                    file_name="clinical_report.txt"
                )

                st.markdown("### 📚 Evidence Sources")
                for i in I[0]:
                    st.info(st.session_state.sources[i])

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

                hospital_results = [st.session_state.documents[i] for i in I[0]]
                hospital_context = "\n\n".join(hospital_results)

                with st.spinner("🧠 Generating hospital clinical intelligence..."):
                    hospital_ai = hospital_clinical_reasoning(query, hospital_context)

                output += "### 🏥 Hospital Clinical Intelligence\n\n" + hospital_ai + "\n\n"

            with st.spinner("🌍 Searching global medical research..."):
                ext = external_research_answer(query)

            output += "### 🌍 Global Medical Research\n\n" + ext.get("answer", "No response")

            st.markdown("### 🧠 Hybrid Clinical Decision Intelligence")
            st.write(output)

            st.download_button(
                "📥 Download Hybrid Report",
                output,
                file_name="hybrid_clinical_report.txt"
            )

# ================== SYSTEM HEALTH ==================
st.sidebar.divider()
st.sidebar.subheader("⚙️ System Health")
st.sidebar.write("Embedding Model: MiniLM-L6-v2")
st.sidebar.write("Vector DB: FAISS")
st.sidebar.write("Global AI: Groq LLaMA")
st.sidebar.write("Indexed Pages:", len(st.session_state.documents))

# ================== QUERY HISTORY ==================
st.sidebar.divider()
st.sidebar.subheader("🕒 Recent Queries")
for q in st.session_state.query_history[-5:]:
    st.sidebar.write("•", q)

# ================== FOOTER ==================
st.divider()
st.caption("MedCopilot Enterprise © Hospital AI Platform | Clinical Decision Intelligence")
