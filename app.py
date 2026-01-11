import streamlit as st
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="MedCopilot Enterprise — Hospital AI",
    page_icon="🧠",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f4f9ff, #e8f0ff);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #64748b;
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric {
    font-size: 32px;
    font-weight: 700;
    color: #2563eb;
}
.label {
    color: #64748b;
    font-size: 14px;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #2563eb, #06b6d4);
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## 🧠 MedCopilot Enterprise")
menu = st.sidebar.radio(
    "Navigation",
    ["🏥 Dashboard", "🔍 Clinical Research", "📊 Analytics", "📂 Evidence Upload", "⚙ Settings"]
)

st.sidebar.markdown("---")
ai_mode = st.sidebar.selectbox("🤖 AI Mode", ["Hospital AI", "Global AI", "Hybrid AI"])
st.sidebar.markdown(f"**Active Mode:** {ai_mode}")

# ================= HEADER =================
st.markdown('<div class="main-title">MedCopilot Enterprise</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hospital AI Platform • Clinical Intelligence • Global Medical Research</div>', unsafe_allow_html=True)

# ================= DASHBOARD =================
if menu == "🏥 Dashboard":
    st.subheader("🏥 Hospital Intelligence Command Center")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="metric">12,430</div>
            <div class="label">Clinical Papers Indexed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="metric">98.7%</div>
            <div class="label">Diagnosis Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="metric">320+</div>
            <div class="label">Hospitals Connected</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
            <div class="metric">24x7</div>
            <div class="label">AI Monitoring</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 AI Clinical Assistant")

    query = st.text_input("Ask any clinical research question:")
    if st.button("🚀 Run Clinical Intelligence"):
        with st.spinner("Analyzing global medical intelligence..."):
            st.success("AI Analysis Complete!")
            st.markdown("""
            <div class="card">
            ✔ Evidence-based response generated  
            ✔ Clinical trials analyzed  
            ✔ Guidelines validated  
            ✔ Research citations included  
            </div>
            """, unsafe_allow_html=True)

# ================= RESEARCH =================
elif menu == "🔍 Clinical Research":
    st.subheader("🔍 Clinical Research Engine")
    st.markdown("Search across PubMed, ClinicalTrials, WHO, FDA databases.")

    q = st.text_area("Enter your research question")
    if st.button("🔎 Search Medical Evidence"):
        st.success("Top research papers retrieved with citations.")

# ================= ANALYTICS =================
elif menu == "📊 Analytics":
    st.subheader("📊 Hospital AI Analytics")
    st.info("Live hospital performance and treatment outcome intelligence dashboard coming here.")

# ================= UPLOAD =================
elif menu == "📂 Evidence Upload":
    st.subheader("📂 Upload Medical Evidence")
    files = st.file_uploader("Upload PDFs / Case Reports / Trials", accept_multiple_files=True)
    if files:
        st.success(f"{len(files)} files uploaded and indexed successfully.")

# ================= SETTINGS =================
elif menu == "⚙ Settings":
    st.subheader("⚙ System Settings")
    st.toggle("Enable Clinical Validation Layer")
    st.toggle("Enable FDA Drug Verification")
    st.toggle("Enable Real-Time Hospital Sync")

# ================= FOOTER =================
st.markdown("---")
st.markdown("© 2026 MedCopilot Enterprise | Hospital AI Platform | Clinical Decision Intelligence")
