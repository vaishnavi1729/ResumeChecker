# ==============================
# VBM – AI-Powered Resume Checker
# Streamlit Version (Deployable)
# ==============================

# 1️⃣ Install dependencies (only needed in Colab, skip if running locally/Streamlit Cloud)
# !pip install streamlit pymupdf python-docx scikit-learn sentence-transformers matplotlib wordcloud numpy

# 2️⃣ Imports
import streamlit as st
import fitz, docx, re, os, json, sqlite3, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 3️⃣ Setup
DB_NAME = "vbm.db"
os.makedirs("uploads", exist_ok=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Common skills dictionary (expandable)
SKILLS_LIST = [
    "python","java","sql","ml","machine learning","cloud","docker","kubernetes",
    "excel","c++","tensorflow","pytorch","aws","azure","linux","git"
]

# 4️⃣ Initialize DB
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS job_description (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        jd_text TEXT,
                        jd_emb BLOB
                     )""")
        c.execute("""CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        resume_name TEXT,
                        score REAL,
                        verdict TEXT,
                        missing_skills TEXT,
                        feedback TEXT
                     )""")
init_db()

# 5️⃣ Helper Functions
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        return " ".join([page.get_text("text") for page in doc])
    elif file_path.lower().endswith(".docx"):
        return " ".join([p.text for p in docx.Document(file_path).paragraphs])
    else:
        return ""

def clean_text(text): 
    return re.sub(r'\s+', ' ', text).strip().lower()

def parse_resume(file_path):
    text = clean_text(extract_text(file_path))
    skills = [s for s in SKILLS_LIST if s in text]
    education = " ".join(re.findall(r"(education|bachelor|master|degree|university|college).*", text))
    experience = " ".join(re.findall(r"(experience|worked|intern|project|developed).*", text))
    projects = " ".join(re.findall(r"(project|built|developed|created).*", text))
    return {
        "Education": education,
        "Skills": " ".join(skills),
        "Experience": experience,
        "Projects": projects,
        "FullText": text
    }

def evaluate_resume(resume_text, jd_text, jd_emb=None, weight_keywords=0.5, weight_semantics=0.5):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    
    # keyword overlap score
    keyword_score = len(resume_words & jd_words)/max(len(jd_words),1)
    
    # semantic similarity
    if jd_emb is None: 
        jd_emb = embedder.encode([jd_text])
    resume_emb = embedder.encode([resume_text])
    
    # Ensure 2D arrays for cosine_similarity
    semantic_score = float(cosine_similarity(
        resume_emb.reshape(1, -1),
        jd_emb.reshape(1, -1)
    )[0][0])
    
    # final weighted score
    final_score = (keyword_score*weight_keywords + semantic_score*weight_semantics)*100
    
    missing_skills = [s for s in SKILLS_LIST if s in jd_text and s not in resume_text]
    
    verdict = "High" if final_score>=75 else "Medium" if final_score>=50 else "Low"
    
    feedback = f"Score: {final_score:.1f}. " + \
               ("✅ Great match!" if verdict=="High" else 
                "⚠️ Partial match." if verdict=="Medium" else 
                "❌ Low alignment.")
    if missing_skills: 
        feedback += f" Missing: {', '.join(missing_skills[:10])}."
    
    return final_score, verdict, missing_skills, feedback

def visualize_resume(resume_json, missing_skills):
    matched_skills = [s for s in resume_json["Skills"].split() if s not in missing_skills]
    
    st.bar_chart({"Matched Skills": len(matched_skills), "Missing Skills": len(missing_skills)})
    
    wc = WordCloud(width=800, height=400, background_color='white').generate(resume_json["FullText"])
    st.image(wc.to_array(), caption="Resume Word Cloud")

# 6️⃣ Streamlit UI
st.title("📄 VBM – AI-Powered Resume Checker")

jd_text = st.text_area("📌 Paste Job Description Here")

resume_file = st.file_uploader("📂 Upload Resume", type=["pdf","docx"])

if st.button("Evaluate") and resume_file and jd_text:
    # Save uploaded resume
    path = os.path.join("uploads", resume_file.name)
    with open(path, "wb") as f: 
        f.write(resume_file.read())
    
    resume_json = parse_resume(path)
    
    # Save JD
    jd_emb = embedder.encode([jd_text])
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO job_description (jd_text,jd_emb) VALUES (?,?)",(jd_text,jd_emb.tobytes()))
        conn.commit()
    
    # Evaluate
    score, verdict, missing_skills, feedback = evaluate_resume(
        resume_json["FullText"], jd_text, jd_emb=jd_emb
    )
    
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO evaluations (resume_name,score,verdict,missing_skills,feedback) VALUES (?,?,?,?,?)",
                  (resume_file.name,score,verdict,json.dumps(missing_skills),feedback))
        conn.commit()
    
    # Display results
    st.subheader("✅ Results")
    st.json({
        "Resume": resume_file.name,
        "Score": round(score,2),
        "Verdict": verdict,
        "Missing Skills": missing_skills,
        "Feedback": feedback
    })
    
    st.subheader("📊 Visualizations")
    visualize_resume(resume_json, missing_skills)

# 7️⃣ Show past evaluations
st.subheader("📂 Previous Evaluations")
with sqlite3.connect(DB_NAME) as conn:
    c = conn.cursor()
    c.execute("SELECT resume_name,score,verdict,feedback FROM evaluations ORDER BY id DESC")
    rows = c.fetchall()

if rows:
    for r in rows:
        st.write(f"**{r[0]}** → Score: {round(r[1],2)} | Verdict: {r[2]}")
        st.caption(r[3])
else:
    st.write("No evaluations yet.")
