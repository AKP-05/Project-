import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- UI --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer")
st.write("Analyze how well your resume matches a job description")

resume_text = st.text_area("Paste Resume Text", height=200)
job_text = st.text_area("Paste Job Description", height=200)

# -------------------- Functions --------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

def extract_keywords(text):
    words = clean_text(text).split()
    return set(words)

# -------------------- Button Action --------------------
if st.button("Analyze Resume"):
    if resume_text.strip() == "" or job_text.strip() == "":
        st.warning("⚠️ Please enter both Resume and Job Description.")
    else:
        # Clean text
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_text)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        # Cosine Similarity
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

        # Keyword comparison
        resume_words = extract_keywords(resume_text)
        job_words = extract_keywords(job_text)
        missing_skills = job_words - resume_words

        # -------------------- Output --------------------
        st.success(f"✅ Resume Match Score: {similarity_score:.2f}%")

        st.subheader("❌ Missing Skills / Keywords")
        if len(missing_skills) == 0:
            st.write("No major skills missing. Resume matches well!")
        else:
            st.write(", ".join(list(missing_skills)[:15]))

        st.subheader("💡 Suggestions")
        st.write("• Add missing skills relevant to the job description.")
        st.write("• Use clear technical keywords.")
        st.write("• Highlight projects and experience matching the role.")