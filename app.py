import streamlit as st
import re
#import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer")

st.write("Analyze your resume against a job description and get improvement suggestions.")

# Input areas
resume_text = st.text_area("📌 Paste Resume Text", height=200)
job_text = st.text_area("📌 Paste Job Description", height=200)

# ---------- Helper Functions ----------

def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower())

def extract_keywords(text):
    doc = nlp(text)
    return set([token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]])

def check_sections(resume):
    sections = {
        "email": bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", resume)),
        "education": "education" in resume.lower(),
        "skills": "skill" in resume.lower(),
        "experience": "experience" in resume.lower()
    }
    return sections

def grammar_warning(resume):
    sentences = resume.split(".")
    long_sentences = [s for s in sentences if len(s.split()) > 35]
    return len(long_sentences)

# ---------- Main Logic ----------

if st.button("🔍 Analyze Resume"):
    if resume_text.strip() == "" or job_text.strip() == "":
        st.warning("Please provide both Resume and Job Description")
    else:
        # Similarity Score
        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform([resume_text, job_text])
        score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

        # Keyword Analysis
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_text)
        missing_skills = sorted(list(job_keywords - resume_keywords))[:10]

        # Section Check
        sections = check_sections(resume_text)

        # Grammar / Quality Check
        grammar_issues = grammar_warning(resume_text)

        # ---------- OUTPUT ----------

        st.success(f"✅ Resume Match Score: {score:.2f}%")

        # Missing Skills
        st.subheader("❌ Missing / Weak Skills")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.write("No major skill gaps found")

        # Missing Sections
        st.subheader("📂 Resume Section Check")
        for section, present in sections.items():
            if present:
                st.write(f"✅ {section.capitalize()} section found")
            else:
                st.write(f"❌ {section.capitalize()} section missing")

        # Grammar & Quality
        st.subheader("✍️ Quality Suggestions")
        if grammar_issues > 0:
            st.write("⚠️ Resume has very long sentences. Consider shortening them.")
        else:
            st.write("✅ Sentence length looks good")

        # Final Suggestions
        st.subheader("📌 Improvement Suggestions")
        suggestions = []

        if score < 60:
            suggestions.append("Add more job-specific skills and keywords.")
        if not sections["skills"]:
            suggestions.append("Add a dedicated Skills section.")
        if not sections["education"]:
            suggestions.append("Mention your Education details clearly.")
        if missing_skills:
            suggestions.append("Include missing skills relevant to the job description.")

        if suggestions:
            for s in suggestions:
                st.write("•", s)
        else:
            st.write("Your resume is well-optimized for this role 🎉")