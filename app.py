import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Analyzer")
st.caption("ML-based Resume Evaluation & Suggestions System")

# ---------------- INPUT ----------------
resume_text = st.text_area("Paste Resume Text", height=220)
job_text = st.text_area("Paste Job Description", height=220)

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    return text.lower()

def get_keywords(text):
    return set(clean_text(text).split())

# ---------------- ANALYSIS ----------------
if st.button("Analyze Resume"):
    if resume_text.strip() == "" or job_text.strip() == "":
        st.warning("⚠️ Please enter both resume and job description.")
    else:
        # Clean texts
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_text)

        # ML MODEL (TF-IDF)
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        # Similarity Score
        score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

        # Keyword Comparison
        resume_words = get_keywords(resume_text)
        job_words = get_keywords(job_text)
        missing = job_words - resume_words

        # ---------------- OUTPUT ----------------
        st.success(f"✅ Resume Match Score: {score:.2f}%")

        st.subheader("❌ Missing Skills / Keywords")
        if not missing:
            st.write("No major skills missing. Resume matches well!")
        else:
            st.write(", ".join(list(missing)[:12]))

        st.subheader("💡 Improvement Suggestions")
        st.markdown("""
        • Add missing technical skills mentioned in job description  
        • Highlight relevant projects and internships  
        • Use industry-standard keywords  
        • Keep resume concise and role-focused  
        """)

        