import streamlit as st
import os, json, pickle, datetime
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from external_research import external_research_answer

# ======================================================
# CONFIG
# ======================================================
st.set_page_config("ĀROGYABODHA AI — Clinical Research Copilot","🧠",layout="wide")
st.info("ℹ️ Clinical research support only. Not for diagnosis.")

PDF_FOLDER="medical_library"
VECTOR_FOLDER="vector_cache"
INDEX_FILE=f"{VECTOR_FOLDER}/index.faiss"
CACHE_FILE=f"{VECTOR_FOLDER}/cache.pkl"
ANALYTICS_FILE="analytics_log.json"
FDA_DB="fda_registry.json"

os.makedirs(PDF_FOLDER,exist_ok=True)
os.makedirs(VECTOR_FOLDER,exist_ok=True)

# ======================================================
# SESSION
# ======================================================
for k,v in {
    "index":None,
    "documents":[],
    "sources":[],
    "index_ready":False,
    "role":"Doctor"
}.items():
    st.session_state.setdefault(k,v)

# ======================================================
# HEADER
# ======================================================
c1,c2=st.columns([6,1])
with c1:
    st.markdown("## 🧠 ĀROGYABODHA AI")
    st.caption("Evidence-Locked Clinical Research Copilot")
with c2:
    st.session_state.role=st.selectbox("Role",["Doctor","Researcher"])

# ======================================================
# EMBEDDINGS
# ======================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder=load_embedder()

# ======================================================
# FDA DB
# ======================================================
if not os.path.exists(FDA_DB):
    json.dump({
        "temozolomide":"FDA Approved",
        "bevacizumab":"FDA Approved",
        "car-t":"Experimental / Trial"
    },open(FDA_DB,"w"))

FDA_REGISTRY=json.load(open(FDA_DB))

# ======================================================
# UTILITIES
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

def semantic_evidence_level(answer,chunks):
    scores=[semantic_similarity(answer,c) for c in chunks]
    best=max(scores) if scores else 0
    if best>=0.45: return "STRONG",int(best*100)
    if best>=0.20: return "PARTIAL",int(best*100)
    return "NONE",0

def confidence_score(ans,n):
    s=60
    if n>=3: s+=15
    if "fda" in ans.lower(): s+=10
    if any(x in ans.lower() for x in ["survival","outcome"]): s+=10
    return min(s,95)

def extract_outcomes(text):
    return pd.DataFrame([
        {"Treatment":d.title(),"FDA Status":s}
        for d,s in FDA_REGISTRY.items() if d in text.lower()
    ])

def hospital_answer(q,ctx):
    prompt=f"Use ONLY hospital evidence:\n{ctx}\nQuestion:{q}"
    return external_research_answer(prompt).get("answer","")

def generate_report(q,m,a,c,cv,s):
    r=f"""Clinical Research Report
-------------------------
Query: {q}
Mode: {m}
Confidence: {c}%
Evidence Match: {cv}%

Answer:
{a}

Sources:
"""
    for x in s: r+=f"- {x}\n"
    return r

# ======================================================
# BUILD INDEX
# ======================================================
def build_index():
    docs,srcs=[],[]
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            reader=PdfReader(os.path.join(PDF_FOLDER,pdf))
            for i,p in enumerate(reader.pages):
                t=p.extract_text()
                if t and len(t)>100:
                    docs.append(t)
                    srcs.append(f"{pdf} – Page {i+1}")

    if not docs:
        return None,[],[]

    emb=embedder.encode(docs).astype("float32")
    idx=faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)

    faiss.write_index(idx,INDEX_FILE)
    pickle.dump({"documents":docs,"sources":srcs},open(CACHE_FILE,"wb"))

    return idx,docs,srcs

# ======================================================
# LOAD CACHE
# ======================================================
if os.path.exists(INDEX_FILE) and not st.session_state.index_ready:
    st.session_state.index=faiss.read_index(INDEX_FILE)
    data=pickle.load(open(CACHE_FILE,"rb"))
    st.session_state.documents=data["documents"]
    st.session_state.sources=data["sources"]
    st.session_state.index_ready=True

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.subheader("📁 Medical Evidence Library")

uploads=st.sidebar.file_uploader("Upload PDFs",type=["pdf"],accept_multiple_files=True)
if uploads:
    for f in uploads:
        open(os.path.join(PDF_FOLDER,f.name),"wb").write(f.getbuffer())
    st.sidebar.success("PDFs uploaded")

if st.sidebar.button("🔄 Build Index"):
    st.session_state.index,st.session_state.documents,st.session_state.sources=build_index()
    st.session_state.index_ready=True
    st.sidebar.write("Indexed pages:",len(st.session_state.documents))

st.sidebar.subheader("🕒 Recent Queries")
if os.path.exists(ANALYTICS_FILE):
    for q in json.load(open(ANALYTICS_FILE))[-5:][::-1]:
        st.sidebar.write(f"• {q['query']} ({q['mode']})")

# ======================================================
# QUERY
# ======================================================
query=st.text_input("Ask clinical research question")
mode=st.radio("AI Mode",["Hospital AI","Global AI","Hybrid AI"],horizontal=True)

if st.button("🚀 Analyze") and query:

    if not st.session_state.index_ready:
        st.warning("Upload PDFs and build index first")
        st.stop()

    log_query(query,mode)

    t1,t2,t3,t4=st.tabs(["🏥 Hospital","🌍 Global","🧪 Outcomes","📚 Library"])

    if mode in ["Hospital AI","Hybrid AI"]:
        qemb=embedder.encode([query]).astype("float32")
        _,I=st.session_state.index.search(qemb,5)

        chunks=[st.session_state.documents[i] for i in I[0]]
        context="\n\n".join(chunks)

        raw=hospital_answer(query,context)

        level,coverage=semantic_evidence_level(raw,chunks)
        conf=confidence_score(raw,len(chunks))
        srcs=[st.session_state.sources[i] for i in I[0]]

        with t1:
            st.metric("Confidence",f"{conf}%")
            st.metric("Evidence Match",f"{coverage}%")

            if level=="STRONG": st.success("Strong hospital evidence")
            elif level=="PARTIAL": st.warning("Partial hospital evidence")
            else: st.error("No sufficient hospital evidence")

            st.write(raw)
            for s in srcs: st.info(s)

            st.download_button(
                "📥 Download Report",
                generate_report(query,mode,raw,conf,coverage,srcs),
                file_name="clinical_report.txt"
            )

        with t3:
            df=extract_outcomes(raw)
            if not df.empty:
                st.table(df)

    if mode in ["Global AI","Hybrid AI"]:
        with t2:
            st.write(external_research_answer(query).get("answer",""))

    with t4:
        for pdf in os.listdir(PDF_FOLDER):
            if pdf.endswith(".pdf"):
                c1,c2=st.columns([8,1])
                with c1: st.write("📄",pdf)
                with c2:
                    if st.button("🗑️",key=pdf):
                        os.remove(os.path.join(PDF_FOLDER,pdf))
                        if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)
                        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
                        st.session_state.index_ready=False
                        st.experimental_rerun()

# ======================================================
# FOOTER
# ======================================================
st.caption("ĀROGYABODHA AI • Evidence-Locked Clinical Research Intelligence")
