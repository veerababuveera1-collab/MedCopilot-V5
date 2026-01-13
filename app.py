import streamlit as st
import os, json, pickle, datetime
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer
from Bio import Entrez

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="ĀROGYABODHA AI — Clinical Research Copilot",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# DISCLAIMER (MANDATORY)
# ======================================================
st.info(
    "ℹ️ ĀROGYABODHA AI is a clinical research decision-support system only. "
    "It does NOT provide diagnosis or treatment. "
    "Final clinical decisions must be made by licensed medical professionals."
)

# ======================================================
# PATHS
# ======================================================
PDF_FOLDER = "medical_library"
VECTOR_FOLDER = "vector_cache"
INDEX_FILE = f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE = f"{VECTOR_FOLDER}/cache.pkl"
ANALYTICS_FILE = "analytics_log.json"
FDA_DB = "fda_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "index": None,
    "documents": [],
    "sources": [],
    "index_ready": False,
    "show_help": False,
    "show_analytics": False,
    "role": "Doctor"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# HEADER
# ======================================================
c1, c2, c3, c4 = st.columns([6,1,1,1])
with c1:
    st.markdown("## 🧠 ĀROGYABODHA AI")
    st.caption("Evidence-Locked • Semantic-Validated • Clinical Research Copilot")
with c2:
    if st.button("❓ Help"):
        st.session_state.show_help = not st.session_state.show_help
with c3:
    st.session_state.role = st.selectbox("Role", ["Doctor","Researcher"])
with c4:
    if st.button("📊 Analytics"):
        st.session_state.show_analytics = not st.session_state.show_analytics

# ======================================================
# HELP PANEL
# ======================================================
if st.session_state.show_help:
    st.markdown("""
### ℹ️ How ĀROGYABODHA AI Works

**AI Modes**
- 🏥 Hospital AI → Hospital PDFs only (evidence-locked)
- 🌍 Global AI → PubMed & global research
- 🔀 Hybrid AI → Compare both (separate outputs)

**Safety**
- Semantic validation checks meaning
- Partial evidence → cautious summary
- No evidence → answer blocked

**Roles**
- Doctor → Conservative summaries
- Researcher → Detailed comparisons

**Example**
Query: *Glioblastoma treatments >60*
""")

# ======================================================
# EMBEDDING MODEL
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

# ======================================================
# FDA REGISTRY
# ======================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide":"FDA Approved",
        "bevacizumab":"FDA Approved",
        "car-t":"Experimental / Trial Only"
    }, open(FDA_DB,"w"))
FDA_REGISTRY = json.load(open(FDA_DB))

# ======================================================
# PUBMED INGESTION
# ======================================================
Entrez.email = "your_email@example.com"

def fetch_pubmed(query, max_results=5):
    try:
        search = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        ids = Entrez.read(search)["IdList"]
        if not ids: return ""
        fetch = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        return fetch.read()
    except:
        return ""

# ======================================================
# HELPERS
# ======================================================
def log_query(q,m):
    logs=[]
    if os.path.exists(ANALYTICS_FILE):
        logs=json.load(open(ANALYTICS_FILE))
    logs.append({"query":q,"mode":m,"time":str(datetime.datetime.now())})
    json.dump(logs,open(ANALYTICS_FILE,"w"),indent=2)

def semantic_similarity(a,b):
    ea=embedder.encode([a])[0]
    eb=embedder.encode([b])[0]
    return float(np.dot(ea,eb)/(np.linalg.norm(ea)*np.linalg.norm(eb)))

def evidence_level(ans,ctx):
    s=semantic_similarity(ans,ctx)
    if s>=0.55: return "STRONG",int(s*100)
    elif s>=0.25: return "PARTIAL",int(s*100)
    else: return "NONE",0

def confidence(ans,n):
    c=60
    if n>=3: c+=15
    if "fda" in ans.lower(): c+=10
    if any(x in ans.lower() for x in ["survival","mortality","outcome"]): c+=10
    return min(c,95)

def extract_outcomes(txt):
    rows=[]
    for d,s in FDA_REGISTRY.items():
        if d in txt.lower():
            rows.append({"Treatment":d.title(),"FDA Status":s})
    return pd.DataFrame(rows)

# ======================================================
# HOSPITAL AI PROMPT
# ======================================================
def hospital_answer(q,ctx):
    prompt=f"""
You are a Hospital Clinical Decision Support AI.

RULES:
- Use ONLY hospital evidence
- No external knowledge
- If insufficient, clearly refuse

Hospital Evidence:
{ctx}

Query:
{q}
"""
    return external_research_answer(prompt).get("answer","")

# ======================================================
# BUILD / LOAD INDEX
# ======================================================
def build_index():
    docs,srcs=[],[]
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            r=PdfReader(os.path.join(PDF_FOLDER,pdf))
            for i,p in enumerate(r.pages[:200]):
                t=p.extract_text()
                if t and len(t)>100:
                    docs.append(t)
                    srcs.append(f"{pdf} – Page {i+1}")
    if not docs: return None,[],[]
    emb=embedder.encode(docs)
    idx=faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb))
    faiss.write_index(idx,INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs},open(CACHE_FILE,"wb"))
    return idx,docs,srcs

if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index=faiss.read_index(INDEX_FILE)
    data=pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents=data["documents"]
    st.session_state.sources=data["sources"]
    st.session_state.index_ready=True

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.subheader("📁 Medical Library")
ups=st.sidebar.file_uploader("Upload PDFs",type=["pdf"],accept_multiple_files=True)
if ups:
    for f in ups:
        open(os.path.join(PDF_FOLDER,f.name),"wb").write(f.getbuffer())
    st.sidebar.success("Uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index,st.session_state.documents,st.session_state.sources=build_index()
    st.session_state.index_ready=True

st.sidebar.divider()
st.sidebar.subheader("🕒 Recent Queries")
if os.path.exists(ANALYTICS_FILE):
    for q in json.load(open(ANALYTICS_FILE))[-5:][::-1]:
        st.sidebar.write(f"• {q['query']} ({q['mode']})")

# ======================================================
# ANALYTICS DASHBOARD
# ======================================================
if st.session_state.show_analytics and os.path.exists(ANALYTICS_FILE):
    st.markdown("## 📊 Analytics Dashboard")
    df=pd.DataFrame(json.load(open(ANALYTICS_FILE)))
    df["time"]=pd.to_datetime(df["time"])
    c1,c2,c3=st.columns(3)
    c1.metric("Total Queries",len(df))
    c2.metric("Hospital AI",(df["mode"]=="Hospital AI").sum())
    c3.metric("Global AI",(df["mode"]=="Global AI").sum())
    st.bar_chart(df["mode"].value_counts())
    st.line_chart(df.groupby(df["time"].dt.date).size())

# ======================================================
# QUERY
# ======================================================
query=st.text_input("Ask a clinical research question")
mode=st.radio("AI Mode",["Hospital AI","Global AI","Hybrid AI"],horizontal=True)
run=st.button("🚀 Analyze")

# ======================================================
# EXECUTION
# ======================================================
if run and query:
    log_query(query,mode)
    t1,t2,t3,t4=st.tabs(["🏥 Hospital","🌍 Global","🧪 Outcomes","📚 Library"])

    if mode in ["Hospital AI","Hybrid AI"]:
        qemb=embedder.encode([query])
        _,I=st.session_state.index.search(np.array(qemb),5)
        ctx="\n\n".join([st.session_state.documents[i] for i in I[0]])
        ans=hospital_answer(query,ctx)
        lvl,cov=evidence_level(ans,ctx)
        conf=confidence(ans,len(I[0]))
        src=[st.session_state.sources[i] for i in I[0]]

        with t1:
            st.metric("Confidence",f"{conf}%")
            st.metric("Evidence Coverage",f"{cov}%")
            if lvl=="STRONG": st.success(ans)
            elif lvl=="PARTIAL": st.warning(ans)
            else: st.error("Insufficient hospital evidence.")
            for s in src: st.info(s)

        with t3:
            df=extract_outcomes(ans)
            if not df.empty: st.table(df)

    if mode in ["Global AI","Hybrid AI"]:
        with t2:
            pub=fetch_pubmed(query)
            prompt=f"Use PubMed abstracts:\n{pub}\nQuestion:{query}"
            st.write(external_research_answer(prompt).get("answer",""))

    with t4:
        for pdf in os.listdir(PDF_FOLDER):
            if pdf.endswith(".pdf"):
                c1,c2=st.columns([8,1])
                c1.write("📄 "+pdf)
                if c2.button("🗑️",key=pdf):
                    os.remove(os.path.join(PDF_FOLDER,pdf))
                    if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
                    if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                    st.session_state.index_ready=False
                    st.experimental_rerun()

st.caption("ĀROGYABODHA AI © FINAL • Clinical Research Safe")
