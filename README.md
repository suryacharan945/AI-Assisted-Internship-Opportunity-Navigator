# 🚀 AI-Assisted Internship Opportunity Navigator

🔗 **Live Application:**
👉 [https://ai-assisted-internship-opportunity-navigator-qsa7rwbvgoltzvarx.streamlit.app/](https://ai-assisted-internship-opportunity-navigator-qsa7rwbvgoltzvarx.streamlit.app/)

---

## 📌 Problem Statement

Students often struggle to:

* Find **relevant internships** based on their skills and interests
* Understand **which skills they are missing** for a target role
* Track **real-time opportunities** like internships, hackathons, and learning resources

Most platforms provide listings but **lack personalization, explainability, and decision support**.

---

## 💡 Solution Overview

**AI-Assisted Internship Opportunity Navigator** is an **explainable AI dashboard** that helps students:

* Discover **personalized internship opportunities**
* Understand **why an opportunity was recommended**
* Identify **skill gaps** and get **learning recommendations**
* Stay updated with **real-time internships, hackathons, and courses**

The system is designed to be **simple, transparent, and actionable**.

---

## 🧠 Key Features

### 🎯 1. Opportunity Navigator

* Manual skill & interest input
* Optional **resume upload (PDF)** with automatic skill extraction
* AI-based ranking using:

  * Skill relevance
  * Deadline urgency
* Clear explanations for every recommendation
* Downloadable **PDF recommendation report**

### 🧠 2. Skill Gap Analysis & Learning Recommendations

* Role-based skill comparison
* Identification of missing skills
* Suggested learning platforms for each skill
* **Career Readiness Score (0–100)**
* Downloadable **Skill Gap Report**

### 🌍 3. Real-Time Internship & Learning Updates

* Live updates from public sources
* Role-based filtering
* Covers:

  * Internships & careers
  * Hackathons & events
  * Learning & courses

---

## ⚙️ How the AI Works (In Simple Terms)

* Uses **TF-IDF vectorization** to represent skills and opportunity descriptions
* Applies **cosine similarity** to measure skill relevance
* Combines relevance with **deadline urgency** to compute a final score

**Final Score Formula:**

```
Final Score = Skill Match Weight × Skill Similarity
            + Urgency Weight × Deadline Urgency
```

This ensures recommendations are both **relevant and time-sensitive**.

---

## 📊 Explainability (Why This Is Important)

Unlike black-box recommenders, this system:

* Shows **matched skills**
* Displays **individual scores**
* Explains **why each opportunity appears**
* Helps users make **informed career decisions**

---

## 🛠️ Tech Stack

* **Frontend & App Framework:** Streamlit
* **Language:** Python
* **ML & Data:** scikit-learn, pandas
* **Model Persistence:** joblib
* **Resume Parsing:** PyPDF2
* **Reports:** FPDF
* **Live Updates:** feedparser

---

## 📂 Project Structure

```
├── app.py
├── processed_opportunity_data.csv
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment

The application is deployed on **Streamlit Community Cloud** and accessible via a public URL:

👉 [https://ai-assisted-internship-opportunity-navigator-qsa7rwbvgoltzvarx.streamlit.app/](https://ai-assisted-internship-opportunity-navigator-qsa7rwbvgoltzvarx.streamlit.app/)

---

## 🧪 How to Run Locally (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏁 Final Notes

This project focuses on:

* **Practical usefulness**
* **Explainable AI**
* **Career decision support**
