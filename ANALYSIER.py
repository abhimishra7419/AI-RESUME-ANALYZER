# AI Resume Analyzer 
# Frist step is open terminal and intsall Required libraries
# Run this : pip install nltk scikit-learn numpy pandas

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')



# Helper Functions




def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"\+?\d[\d -]{8,}\d", text)
    return match.group(0) if match else "Not found"

def extract_name(text):
    lines = text.strip().split("\n")
    return lines[0].strip() if len(lines) > 0 else "Not found"

def clean_text(text):
    text = text.lower()
    sw = set(stopwords.words("english"))
    words = [w for w in re.findall(r"\b[a-zA-Z]+\b", text) if w not in sw]
    return " ".join(words)

def extract_skills(text):
    predefined_skills = [
        "python", "java", "c", "c++", "html", "css", "javascript",
        "machine learning", "deep learning", "sql", "excel",
        "data analysis", "communication", "teamwork", "leadership"
    ]
    
    found = []
    text_lower = text.lower()
    for skill in predefined_skills:
        if skill in text_lower:
            found.append(skill)
    return list(set(found))




# Resume Scoring




def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = ((vectors * vectors.T).toarray())[0, 1]
    return round(similarity * 100, 2)




# Main Analyzer Function




def analyze_resume(resume_text, job_description):
    clean_resume = clean_text(resume_text)

    details = {
        "Name": extract_name(resume_text),
        "Email": extract_email(resume_text),
        "Phone": extract_phone(resume_text),
        "Skills Found": extract_skills(resume_text),
        "Match Score (%)": calculate_similarity(clean_resume, clean_text(job_description))
    }

    # Suggestions
    suggestions = []
    if details["Match Score (%)"] < 50:
        suggestions.append("Add more keywords from the job description.")
    if len(details["Skills Found"]) < 3:
        suggestions.append("Add more technical and soft skills.")
    if details["Email"] == "Not found":
        suggestions.append("Mention a professional email on your resume.")
    
    return details, suggestions




resume = """
Abhishek Mishra
Email: abhimishra7419@gmail.com
Phone: +91 9315981823

I am a B.Tech CSE student skilled in 
phython, Machine Learing, HTML, CSS.
Worked on projects in AI and data analysis.
"""

job_description = """
We are looking for a python developer with
skills in machine learning.
data analysis, SQL, teamwork and
communication skills.
"""

result, suggestions = analyze_resume(resume, job_description)

print("=== Resume Analysis Result ===")
for k, v in result.items():
    print(f"{k}: {v}")

print("\n=== Suggestions ===")
for s in suggestions:
    print("- " + s)