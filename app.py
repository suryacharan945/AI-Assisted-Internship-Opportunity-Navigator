# ------------------------------------------------------------
# SAFE OPTIONAL IMPORTS (GLOBAL)
# ------------------------------------------------------------
try:
    import feedparser
except ImportError:
    feedparser = None

# ============================================================
# GLOBAL LOAD (REQUIRED FOR ALL PAGES)
# ============================================================

import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from fpdf import FPDF
import streamlit as st
# ------------------------------------------------------------
# APP TITLE & DESCRIPTION (GLOBAL HEADER)
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Internship & Opportunity Navigator",
    layout="wide"
)

st.title("🚀 AI-Assisted Internship & Opportunity Navigator")
st.markdown(
    """
    **An explainable AI dashboard** that helps students discover relevant internships,  
    identify skill gaps with learning guidance, and stay updated with real-time opportunities.
    """
)

st.markdown("---")


@st.cache_resource
def load_resources():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = pd.read_csv("processed_opportunity_data.csv")
    return vectorizer, df

vectorizer, df = load_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_pdf_report(title, paragraphs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(5)
    for p in paragraphs:
        pdf.multi_cell(0, 8, p)
        pdf.ln(2)
    return pdf.output(dest="S").encode("latin-1")

tab1, tab2, tab3 = st.tabs([
    "🎯 Opportunity Navigator",
    "🧠 Skill Gap & Learning",
    "🌍 Real-Time Updates"
])


with tab1:

    st.title("🎯 Opportunity Navigator")
    st.caption(
        "Personalized recommendations based on skills, interests, and deadline urgency "
        "with clear AI explanations for every decision."
    )

    # --------------------------------------------------------
    # STUDENT PROFILE (LEFT PANEL STYLE)
    # --------------------------------------------------------
    st.subheader("👤 Student Profile")

    skills_input = st.text_input(
        "Enter your skills (comma-separated)",
        "python, machine learning, data science"
    )

    interests_input = st.text_input(
        "Enter your interests / domain",
        "artificial intelligence"
    )

    year_input = st.selectbox(
        "Select your academic year",
        df["eligible_year"].unique()
    )

    top_n = st.slider(
        "Number of recommendations",
        3, 10, 5
    )

    skill_weight = st.slider(
        "Skill relevance weight",
        0.0, 1.0, 0.7
    )

    urgency_weight = 1 - skill_weight

    st.markdown(
        f"""
        **Scoring Formula**  
        Final Score =  
        `{skill_weight:.1f} × Skill Match + {urgency_weight:.1f} × Urgency`
        """
    )

    # --------------------------------------------------------
    # RUN RECOMMENDATION
    # --------------------------------------------------------
    if st.button("🔍 Get Recommendations"):

        # -------- Profile Processing --------
        student_text = clean_text(skills_input + " " + interests_input)
        student_vector = vectorizer.transform([student_text])
        student_skills = set(clean_text(skills_input).split(", "))

        # -------- Deadline Handling --------
        df["deadline"] = pd.to_datetime(df["deadline"])
        df["days_left"] = (df["deadline"] - pd.Timestamp.today()).dt.days
        df["days_left"] = df["days_left"].apply(lambda x: max(x, 0))

        # -------- Similarity & Scoring --------
        opp_vectors = vectorizer.transform(df["combined_text"])
        df["skill_similarity"] = cosine_similarity(
            student_vector, opp_vectors
        ).flatten()

        df["urgency_score"] = 1 - (df["days_left"] / df["days_left"].max())

        df["final_score"] = (
            skill_weight * df["skill_similarity"] +
            urgency_weight * df["urgency_score"]
        )

        results = (
            df[df["eligible_year"] == year_input]
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------
        st.subheader("🏆 Recommended Opportunities")

        report_paragraphs = []

        for _, row in results.iterrows():

            required_skills = set(row["required_skills"].split(", "))
            matched_skills = required_skills.intersection(student_skills)

            with st.expander(
                f"📌 {row['opportunity_title']} ({row['platform']})",
                expanded=True
            ):

                # -------- WHY THIS WAS RECOMMENDED --------
                st.markdown("### 🔍 Why this was recommended?")
                st.markdown(
                    f"""
                    **Matched Skills:** `{', '.join(matched_skills) if matched_skills else 'Limited overlap'}`  
                    **Skill Similarity Score:** `{row['skill_similarity']:.3f}`  
                    **Urgency Score:** `{row['urgency_score']:.3f}`  
                    **Days Left:** `{row['days_left']}`  
                    **Final AI Score:** `{row['final_score']:.3f}`
                    """
                )

                # -------- OPPORTUNITY DETAILS --------
                st.markdown("### 📄 Opportunity Details")
                st.markdown(
                    f"""
                    **Domain:** {row['domain'].title()}  
                    **Required Skills:** {row['required_skills']}  
                    **Opportunity Type:** {row['opportunity_type']}  
                    **Deadline:** {row['deadline'].date()}
                    """
                )

            # -------- REPORT (HUMAN READABLE) --------
            report_paragraphs.append(
                f"""
Opportunity: {row['opportunity_title']} ({row['platform']})

This opportunity was recommended because your profile aligns with the required
skills ({', '.join(matched_skills) if matched_skills else 'partial overlap'})
and matches your interest domain ({row['domain']}).

The system calculated a skill relevance score of {row['skill_similarity']:.2f}
and an urgency score of {row['urgency_score']:.2f}, considering that only
{row['days_left']} days remain before the deadline.

Final Recommendation Score: {row['final_score']:.2f}

Required Skills: {row['required_skills']}
Opportunity Type: {row['opportunity_type']}
Deadline: {row['deadline'].date()}
"""
            )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------
        pdf_bytes = generate_pdf_report(
            "AI Opportunity Recommendation Report",
            report_paragraphs
        )

        st.download_button(
            "⬇️ Download Detailed Recommendation Report",
            pdf_bytes,
            "opportunity_recommendations.pdf",
            "application/pdf"
        )
with tab2:

    st.title("🧠 Skill Gap Analysis & Learning Recommendations")
    st.caption(
        "This section compares your current skills with the skills required for a target role "
        "and recommends what to learn next along with suitable learning platforms."
    )

    # --------------------------------------------------------
    # ROLE → REQUIRED SKILLS (CURATED & EXPLAINABLE)
    # --------------------------------------------------------
    ROLE_SKILL_MAP = {
        "Data Scientist": [
            "python", "statistics", "machine learning",
            "sql", "data visualization", "power bi", "tableau"
        ],
        "Machine Learning Engineer": [
            "python", "machine learning", "deep learning",
            "tensorflow", "pytorch", "model deployment"
        ],
        "Data Analyst": [
            "excel", "sql", "statistics", "power bi", "tableau"
        ],
        "AI Researcher": [
            "machine learning", "deep learning",
            "nlp", "computer vision", "research methodology"
        ]
    }

    LEARNING_RESOURCES = {
        "python": "Coursera / NPTEL / Kaggle",
        "statistics": "Khan Academy / NPTEL",
        "machine learning": "Coursera (Andrew Ng) / Kaggle",
        "deep learning": "DeepLearning.AI / PyTorch Tutorials",
        "sql": "Mode SQL / LeetCode SQL",
        "data visualization": "Tableau Public / Power BI Docs",
        "power bi": "Microsoft Learn",
        "tableau": "Tableau Free Training",
        "tensorflow": "TensorFlow Official Tutorials",
        "pytorch": "PyTorch Official Tutorials",
        "model deployment": "FastAPI + Docker (YouTube)",
        "nlp": "HuggingFace Course",
        "computer vision": "OpenCV Tutorials",
        "research methodology": "Google Scholar + Research Papers"
    }

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------
    st.subheader("🎯 Target Role Selection")

    target_role = st.selectbox(
        "Select the role you are preparing for",
        list(ROLE_SKILL_MAP.keys())
    )

    user_skills_input = st.text_input(
        "Enter your current skills (comma-separated)",
        "python, sql, machine learning"
    )

    user_skills = set(clean_text(user_skills_input).split(", "))
    required_skills = set(ROLE_SKILL_MAP[target_role])

    # --------------------------------------------------------
    # SKILL GAP ANALYSIS
    # --------------------------------------------------------
    st.subheader("📊 Skill Gap Analysis")

    matched_skills = required_skills.intersection(user_skills)
    missing_skills = required_skills - user_skills

    report_paragraphs = []

    if matched_skills:
        st.success("✅ Skills you already have:")
        for skill in sorted(matched_skills):
            st.markdown(f"- **{skill.title()}**")

    if missing_skills:
        st.warning("⚠️ Skills you should learn next:")

        for skill in sorted(missing_skills):
            st.markdown(
                f"""
                **🔹 {skill.title()}**  
                Why it matters: Required for the role of **{target_role}**  
                Recommended learning platforms: *{LEARNING_RESOURCES.get(skill, 'Online platforms')}*
                """
            )

            report_paragraphs.append(
                f"{skill.title()} is a critical skill for the role of {target_role}. "
                f"It is recommended to learn this skill using {LEARNING_RESOURCES.get(skill, 'reliable online resources')}."
            )
    else:
        st.success("🎉 You already have all the core skills required for this role!")

        report_paragraphs.append(
            f"You already possess all the core skills required for the role of {target_role}."
        )

    # --------------------------------------------------------
    # CAREER READINESS SCORE
    # --------------------------------------------------------
    st.subheader("📈 Career Readiness Score")

    readiness_score = int(
        (len(matched_skills) / len(required_skills)) * 100
        if required_skills else 0
    )

    st.metric(
        label=f"Readiness for {target_role}",
        value=f"{readiness_score} / 100"
    )

    if readiness_score >= 80:
        st.success(
            "🎯 You are well-prepared for this role. Focus on applying to opportunities."
        )
    elif readiness_score >= 50:
        st.info(
            "📘 You are moderately prepared. Upskilling in a few areas will strengthen your profile."
        )
    else:
        st.warning(
            "📚 You are in the early stage. Focus on building strong fundamentals."
        )

    report_paragraphs.append(
        f"Overall career readiness score for {target_role} is {readiness_score}/100."
    )

    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------
    st.subheader("⬇️ Download Skill Gap Report")

    pdf_bytes = generate_pdf_report(
        "Skill Gap Analysis & Learning Recommendations Report",
        report_paragraphs
    )

    st.download_button(
        label="Download Skill Gap Report (PDF)",
        data=pdf_bytes,
        file_name="skill_gap_analysis_report.pdf",
        mime="application/pdf"
    )
with tab3:

    st.title("🌍 Real-Time Internship & Learning Updates")
    st.caption(
        "This section provides live updates on internships, hackathons, and learning resources. "
        "You can filter updates based on your target career role."
    )

    # --------------------------------------------------------
    # ROLE FILTER (USER FRIENDLY)
    # --------------------------------------------------------
    role_filter = st.selectbox(
        "🎯 Filter updates by target role",
        [
            "All Roles",
            "Data Scientist",
            "Machine Learning Engineer",
            "Data Analyst",
            "AI Researcher"
        ]
    )

    role_keywords = {
        "Data Scientist": ["data", "science", "analytics", "sql"],
        "Machine Learning Engineer": ["machine", "learning", "ml", "ai"],
        "Data Analyst": ["analysis", "analyst", "dashboard", "excel"],
        "AI Researcher": ["research", "ai", "deep learning", "nlp"]
    }

    # --------------------------------------------------------
    # RSS FEEDS (RELIABLE & PUBLIC)
    # --------------------------------------------------------
    RSS_FEEDS = {
        "💼 Careers & Internships": "https://blog.google/feed/",
        "🏆 Hackathons & Events": "https://devpost.com/feed",
        "📘 Learning & Courses": "https://blog.coursera.org/feed/"
    }

    if feedparser is None:
        st.warning(
            "Real-time updates require the `feedparser` package.\n\n"
            "Install it using: `pip install feedparser`"
        )

    else:
        for section, url in RSS_FEEDS.items():
            feed = feedparser.parse(url)

            with st.expander(section, expanded=True):

                if not feed.entries:
                    st.info("No updates available at the moment.")
                    continue

                shown = 0

                for entry in feed.entries:
                    title = entry.get("title", "").lower()

                    # -------- ROLE-BASED FILTERING --------
                    if role_filter != "All Roles":
                        keywords = role_keywords.get(role_filter, [])
                        if not any(k in title for k in keywords):
                            continue

                    # -------- DISPLAY UPDATE --------
                    st.markdown(
                        f"- **{entry.title}**  \n"
                        f"  🔗 [{entry.link}]({entry.link})"
                    )

                    shown += 1
                    if shown >= 5:
                        break

                if shown == 0:
                    st.info(
                        "No updates matched the selected role. "
                        "Try selecting **All Roles**."
                    )

    # --------------------------------------------------------
    # USER EDUCATION SECTION (IMPORTANT FOR JUDGES)
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("ℹ️ How to Use These Updates")

    st.markdown(
        """
        - **Internships & Careers:** Track industry hiring trends and application announcements  
        - **Hackathons:** Discover competitive events to build projects and gain visibility  
        - **Learning Resources:** Stay updated with new courses and skill development programs  

        These updates help students **act quickly on opportunities** rather than searching manually.
        """
    )


