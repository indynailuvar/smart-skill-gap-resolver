import re
from collections import Counter

import pandas as pd
import streamlit as st

from config import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COURSE_DF_PATH
)

from recommender import (
    recommend_courses,
    load_artifacts,
    get_filter_options
)


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Smart Skill Gap Resolver",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================
# SKILL CLEANING CONFIG
# ======================================================

COURSE_CATEGORY_TERMS = {
    "arts-and-humanities",
    "music-and-art",
    "business",
    "business-strategy",
    "business-essentials",
    "computer-science",
    "software-development",
    "mobile-and-web-development",
    "algorithms",
    "information-technology",
    "data-management",
    "data-science",
    "machine-learning",
    "math-and-logic",
    "probability-and-statistics",
    "physical-science-and-engineering",
    "electrical-engineering",
    "mechanical-engineering",
    "basic-science",
    "life-sciences",
    "health",
    "public-health",
    "patient-care",
    "nutrition",
    "social-sciences",
    "education",
    "psychology",
    "governance-and-society",
    "language-learning",
    "other-languages",
    "learning-english",
    "personal-development",
    "leadership-and-management",
    "finance",
    "marketing",
    "entrepreneurship",
    "cloud-computing",
    "security",
    "computer-security-and-networks",
}

NOISE_SKILL_TERMS = {
    "best",
    "better",
    "bid",
    "bible",
    "biblical",
    "virtue",
    "belief",
    "thought",
    "history",
    "project mine",
    "global",
    "resource",
    "language",
    "process",
    "web",
    "energy",
    "materials",
    "experience",
    "culture",
    "art",
    "music",
    "law",
    "market",
    "project",
    "film",
}


# ======================================================
# CSS STYLE
# ======================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --brown: #675647;
            --peach: #E3BD90;
            --cream: #DFD3B5;
            --stone: #C0BDBC;
            --olive: #6C6E36;
            --ink: #3B3B3B;

            --bg-top: #FAF6EE;
            --bg-bottom: #F1E7D4;
            --paper: #FFFFFF;
            --paper-soft: #FBF6EC;

            --border: rgba(103, 86, 71, 0.14);
            --shadow: 0 16px 40px rgba(103, 86, 71, 0.10);
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", "Segoe UI", "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 6%, rgba(227, 189, 144, 0.35), transparent 30%),
                radial-gradient(circle at 95% 4%, rgba(108, 110, 54, 0.12), transparent 32%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--ink);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        button[kind="header"] {
            display: none;
        }

        h1, h2, h3 {
            font-family: "Fraunces", serif;
            color: var(--brown);
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        p, label, span, div {
            color: var(--ink);
        }

        /* TOP BAR */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }

        .topbar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .topbar-mark {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: var(--olive);
            color: #FFFDF8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }

        .topbar-name {
            font-family: "Fraunces", serif;
            font-weight: 700;
            font-size: 19px;
            color: var(--brown);
            line-height: 1.25;
        }

        .topbar-name span {
            display: block;
            font-family: "Plus Jakarta Sans", sans-serif;
            font-weight: 600;
            font-size: 11px;
            letter-spacing: 0.14em;
            color: var(--stone);
            text-transform: uppercase;
            margin-top: 2px;
        }

        .topbar-tag {
            background: var(--paper-soft);
            border: 1px solid var(--border);
            color: var(--brown);
            border-radius: 999px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }

        /* HERO */
        .hero-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, var(--cream) 0%, var(--peach) 100%);
            border-radius: 28px;
            padding: 40px 44px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(103, 86, 71, 0.18);
            margin-bottom: 24px;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -85px;
            top: -95px;
            background: rgba(103, 86, 71, 0.09);
            border-radius: 999px;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            left: -55px;
            bottom: -75px;
            background: rgba(108, 110, 54, 0.12);
            border-radius: 999px;
        }

        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 760px;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--brown);
            background: rgba(255, 253, 248, 0.65);
            border: 1px solid rgba(103, 86, 71, 0.22);
            border-radius: 999px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 18px;
            letter-spacing: 0.10em;
        }

        .hero-kicker::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--olive);
        }

        .hero-title {
            font-family: "Fraunces", serif;
            color: var(--brown);
            font-size: 42px;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 14px;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            color: var(--ink);
            opacity: 0.92;
            font-size: 15px;
            line-height: 1.85;
        }

        .hero-tags {
            margin-top: 22px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .hero-tag {
            background: var(--paper);
            color: var(--brown);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid rgba(103, 86, 71, 0.18);
        }

        /* METRICS */
        div[data-testid="stMetric"] {
            background: var(--paper);
            border: 1px solid var(--border);
            border-top: 3px solid var(--olive);
            border-radius: 18px;
            padding: 16px 20px;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--stone) !important;
            font-size: 12px !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        div[data-testid="stMetricValue"] {
            font-family: "Fraunces", serif !important;
            color: var(--brown) !important;
            font-weight: 700 !important;
        }

        /* PANEL */
        .panel {
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 26px 28px;
            box-shadow: var(--shadow);
            margin-top: 22px;
            margin-bottom: 18px;
        }

        .panel-title {
            font-family: "Fraunces", serif;
            color: var(--brown);
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 7px;
        }

        .panel-subtitle {
            color: var(--ink);
            opacity: 0.78;
            font-size: 14px;
            line-height: 1.7;
        }

        .note-box {
            background: var(--paper-soft);
            border: 1px solid var(--border);
            border-left: 4px solid var(--olive);
            border-radius: 16px;
            padding: 15px 18px;
            color: var(--ink);
            line-height: 1.75;
            font-size: 13.5px;
            margin-bottom: 18px;
        }

        .note-box b {
            color: var(--brown);
        }

        .selected-box {
            background: var(--paper-soft);
            border: 1px dashed rgba(103, 86, 71, 0.30);
            border-radius: 18px;
            padding: 14px 18px;
            margin-bottom: 16px;
            color: var(--ink);
            line-height: 1.8;
            font-size: 14px;
        }

        .selected-box b {
            color: var(--brown);
        }

        /* INPUT */
        .stTextInput > div > div > input {
            background-color: var(--paper) !important;
            border: 1px solid var(--stone) !important;
            border-radius: 15px !important;
            color: var(--ink) !important;
            min-height: 48px !important;
        }

        .stTextInput > div > div > input::placeholder {
            color: #A39B8E !important;
        }

        .stTextInput > div > div:focus-within,
        .stMultiSelect div[data-baseweb="select"]:focus-within > div,
        .stSelectbox div[data-baseweb="select"]:focus-within > div {
            border-color: var(--olive) !important;
            box-shadow: 0 0 0 3px rgba(108, 110, 54, 0.15) !important;
        }

        .stMultiSelect div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: var(--paper) !important;
            border: 1px solid var(--stone) !important;
            border-radius: 15px !important;
            min-height: 48px !important;
        }

        .stMultiSelect span {
            color: var(--ink) !important;
        }

        .stMultiSelect [data-baseweb="tag"] {
            background-color: var(--cream) !important;
            border-radius: 8px !important;
            border: none !important;
        }

        .stMultiSelect [data-baseweb="tag"] span {
            color: var(--brown) !important;
            font-weight: 700 !important;
        }

        .stSlider {
            padding-top: 8px;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: var(--olive) !important;
            border-color: var(--olive) !important;
        }

        .streamlit-expanderHeader {
            color: var(--brown) !important;
            font-weight: 800 !important;
            background: var(--paper-soft) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
        }

        /* BUTTON */
        .stFormSubmitButton > button {
            width: 100%;
            border: none;
            border-radius: 16px;
            padding: 0.90rem 1rem;
            background: var(--olive);
            color: #FFFDF8;
            font-weight: 800;
            font-size: 15px;
            box-shadow: 0 15px 35px rgba(108, 110, 54, 0.28);
            transition: background-color 0.18s ease;
        }

        .stFormSubmitButton > button:hover {
            background: var(--brown);
            color: #FFFDF8;
        }

        .stDownloadButton > button {
            border-radius: 15px;
            border: 1px solid var(--brown);
            background: var(--paper);
            color: var(--brown);
            font-weight: 800;
            padding: 0.70rem 1rem;
            transition: background-color 0.18s ease;
        }

        .stDownloadButton > button:hover {
            background: var(--cream);
            color: var(--brown);
            border-color: var(--brown);
        }

        /* TABLE */
        .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 22px;
            overflow: hidden;
            box-shadow: var(--shadow);
            background: var(--paper);
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid var(--border);
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 31px;
            }

            .hero-card {
                padding: 28px 26px;
            }

            .topbar {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def check_model_files():
    missing_files = []

    for path in [TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH, COURSE_DF_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    return missing_files


def render_topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-brand">
                <div class="topbar-mark">🎓</div>
                <div class="topbar-name">
                    Smart Skill Gap Resolver
                    <span>Coursera Course Recommender</span>
                </div>
            </div>
            <div class="topbar-tag">TF-IDF · Cosine Similarity Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-content">
                <div class="hero-kicker">COURSERA-BASED COURSE RECOMMENDER</div>
                <div class="hero-title">Smart Skill Gap Resolver</div>
                <div class="hero-subtitle">
                    Sistem rekomendasi course untuk membantu karyawan dan HR memilih pelatihan
                    yang relevan berdasarkan skill target yang tersedia pada dataset Coursera.
                    Input dibuat berbasis data agar hasil rekomendasi lebih valid, rapi, dan mudah dipertanggungjawabkan.
                </div>
                <div class="hero-tags">
                    <span class="hero-tag">TF-IDF</span>
                    <span class="hero-tag">Cosine Similarity</span>
                    <span class="hero-tag">Content-Based Filtering</span>
                    <span class="hero-tag">Skill Matching</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_panel(title, subtitle):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def normalize_skill_name(skill):
    if pd.isna(skill):
        return None

    skill = str(skill).strip()

    if not skill:
        return None

    skill = re.sub(r"\([^)]*\)", "", skill)
    skill = skill.replace("&", " and ")
    skill = re.sub(r"[^A-Za-z0-9\+\#\.\s-]", " ", skill)
    skill = re.sub(r"\s+", " ", skill).strip()

    if not skill:
        return None

    for term in sorted(COURSE_CATEGORY_TERMS, key=len, reverse=True):
        pattern = rf"\b{re.escape(term)}\b"
        skill = re.sub(pattern, "", skill, flags=re.IGNORECASE).strip()

    skill = re.sub(r"\s+", " ", skill).strip()

    if not skill:
        return None

    skill_lower = skill.lower()

    if skill_lower in NOISE_SKILL_TERMS:
        return None

    if len(skill) < 3:
        return None

    if len(skill.split()) > 5:
        return None

    if len(skill) > 50:
        return None

    alphabet_count = sum(char.isalpha() for char in skill)
    if alphabet_count < 3:
        return None

    return skill.title()


def split_skills_from_dataset(skill_text):
    if pd.isna(skill_text):
        return []

    skill_text = str(skill_text).strip()

    if not skill_text:
        return []

    parts = re.split(r"\s{2,}|[,;|]+", skill_text)

    cleaned_skills = []

    for part in parts:
        skill = normalize_skill_name(part)

        if skill:
            cleaned_skills.append(skill)

    return cleaned_skills


def extract_unique_skills(df, min_frequency=3, max_options=1000):
    skill_counter = Counter()

    for skill_text in df["skills"].fillna(""):
        skills = split_skills_from_dataset(skill_text)

        for skill in skills:
            skill_counter[skill] += 1

    valid_skills = [
        skill
        for skill, count in skill_counter.items()
        if count >= min_frequency
    ]

    valid_skills = sorted(
        valid_skills,
        key=lambda skill: (-skill_counter[skill], skill)
    )

    return valid_skills[:max_options]


def prepare_recommendation_table(results: pd.DataFrame) -> pd.DataFrame:
    display_df = results.copy()

    display_df["similarity_score"] = display_df["similarity_score"].round(3)
    display_df["final_score"] = display_df["final_score"].round(3)
    display_df["rating"] = display_df["rating"].round(2)

    recommendation_table = display_df[
        [
            "course_name",
            "organization",
            "difficulty",
            "certificate_type",
            "rating",
            "similarity_score",
            "final_score"
        ]
    ].rename(
        columns={
            "course_name": "Course Name",
            "organization": "Organization",
            "difficulty": "Level",
            "certificate_type": "Certificate Type",
            "rating": "Rating",
            "similarity_score": "Similarity",
            "final_score": "Score"
        }
    )

    return recommendation_table


def render_selected_skills(selected_skills):
    if not selected_skills:
        return

    selected_text = ", ".join(selected_skills)

    st.markdown(
        f"""
        <div class="selected-box">
            <b>Skill target yang dipilih:</b><br>
            {selected_text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# LOAD APP
# ======================================================

inject_css()

missing_files = check_model_files()

if missing_files:
    st.error("Model belum dibuat. Jalankan perintah berikut di terminal:")
    st.code("python src/model.py", language="bash")

    st.write("File yang belum ditemukan:")
    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


df, _, _ = load_artifacts()
filter_options = get_filter_options()

skill_options = extract_unique_skills(
    df,
    min_frequency=3,
    max_options=1000
)


# ======================================================
# HEADER
# ======================================================

render_topbar()
render_hero()


# ======================================================
# METRICS
# ======================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Courses", f"{len(df):,}")

with col2:
    st.metric("Valid Skills", f"{len(skill_options):,}")

with col3:
    st.metric("Average Rating", round(df["rating"].mean(), 2))

with col4:
    st.metric("Difficulty Levels", df["difficulty"].nunique())


# ======================================================
# FORM
# ======================================================

render_panel(
    "Course Recommendation Form",
    "Pilih skill target yang tersedia pada dataset Coursera. Sistem akan mencocokkan input dengan konten course menggunakan pendekatan Content-Based Filtering."
)

st.markdown(
    """
    <div class="note-box">
        <b>Catatan validasi:</b> Skill tidak diketik bebas. Skill dipilih dari data unik pada kolom
        <b>Skills</b> di dataset Coursera agar model hanya memproses istilah yang memang tersedia di data.
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("recommendation_form"):
    employee_name = st.text_input(
        "Nama Karyawan",
        placeholder="Contoh: Indy Nailuvar"
    )

    selected_skills = st.multiselect(
        "Pilih Skill Target dari Dataset Coursera",
        options=skill_options,
        placeholder="Cari skill valid, contoh: Python Programming, Data Analysis, Machine Learning"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        selected_levels = st.multiselect(
            "Filter Level",
            options=filter_options["difficulty"],
            default=["All"]
        )

    with col_b:
        selected_certificates = st.multiselect(
            "Filter Certificate Type",
            options=filter_options["certificate_type"],
            default=["All"]
        )

    with col_c:
        top_n = st.slider(
            "Jumlah Rekomendasi",
            min_value=3,
            max_value=15,
            value=5
        )

    with st.expander("Filter tambahan"):
        selected_organizations = st.multiselect(
            "Filter Organization",
            options=filter_options["organization"],
            default=["All"]
        )

        min_rating = st.slider(
            "Minimum Rating",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )

        min_similarity = st.slider(
            "Minimum Similarity",
            min_value=0.00,
            max_value=1.00,
            value=0.01,
            step=0.01
        )

    submitted = st.form_submit_button("Cari Rekomendasi Course")


# ======================================================
# RESULT
# ======================================================

if submitted:
    if not selected_skills:
        st.warning("Pilih minimal satu skill target terlebih dahulu.")

    else:
        input_skill_text = ", ".join(selected_skills)

        results = recommend_courses(
            input_skill_text=input_skill_text,
            role_text="",
            top_n=top_n,
            level_filter=selected_levels,
            certificate_filter=selected_certificates,
            organization_filter=selected_organizations,
            min_rating=min_rating,
            min_similarity=min_similarity
        )

        if results.empty:
            st.error("Tidak ada course yang cocok dengan skill dan filter yang dipilih.")

        else:
            if employee_name.strip():
                st.success(f"Rekomendasi course berhasil dibuat untuk {employee_name}.")
            else:
                st.success("Rekomendasi course berhasil dibuat.")

            render_panel(
                "Hasil Course yang Direkomendasikan",
                "Berikut adalah daftar course terbaik berdasarkan skill target dan filter yang dipilih."
            )

            render_selected_skills(selected_skills)

            recommendation_table = prepare_recommendation_table(results)

            st.dataframe(
                recommendation_table,
                use_container_width=True,
                hide_index=True
            )

            csv = recommendation_table.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Hasil Rekomendasi",
                data=csv,
                file_name="hasil_rekomendasi_course.csv",
                mime="text/csv"
            )
