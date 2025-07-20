# 🤖 Automated Resume Screening using NLP

A smart recruitment assistant that uses Natural Language Processing (NLP) to analyze resumes and suggest suitable job roles based on candidate skills, experience, and education.

---

## 📘 About

This project automates the initial resume screening process by extracting text from resumes (PDF/DOCX), applying NLP techniques, and comparing candidate qualifications against predefined job role descriptions. It helps streamline hiring by identifying the most relevant roles for a given candidate.

---

## 🔧 Features

- 📄 Resume parsing (PDF/DOCX)
- 🧠 NLP-based text analysis (TF-IDF, NER, etc.)
- 🔍 Matching with job descriptions using cosine similarity
- 🧾 Suggests top-fit roles for the candidate
- 📊 Optional dashboard or web interface using Django

---

## 💡 Technologies Used

| Category         | Tools/Libraries                          |
|------------------|------------------------------------------|
| Language         | Python                                   |
| NLP              | spaCy, NLTK, scikit-learn                |
| Vectorization    | TF-IDF, CountVectorizer                  |
| Web Framework    | Django / Streamlit (optional)            |
| Frontend         | HTML, CSS, JavaScript                    |
| Parsing Resumes  | PyMuPDF / python-docx                    |
| Others           | NumPy, Pandas                            |




 Sample Use Case
 
Upload a resume.

The system parses and processes the text.

It calculates similarity with stored job descriptions.

Displays top 3 matching job roles with similarity scores.

📈 Future Enhancements
Integrate real-time job feeds (LinkedIn/Indeed)

Add chatbot interface for interaction

Skill gap analysis with upskilling suggestions

Admin dashboard for HR managers


