import joblib
import pandas as pd
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COURSE_DF_PATH
)
from preprocessing import clean_text


@lru_cache(maxsize=1)
def load_artifacts():
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    df = joblib.load(COURSE_DF_PATH)

    return df, vectorizer, tfidf_matrix


def normalize_rating(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    max_rating = series.max()

    if max_rating <= 5:
        return (series / 5).clip(0, 1)

    min_rating = series.min()

    if max_rating == min_rating:
        return series * 0

    return (series - min_rating) / (max_rating - min_rating)


def is_all_filter(filter_values):
    if not filter_values:
        return True

    if isinstance(filter_values, str):
        return filter_values.lower() == "all"

    return "All" in filter_values


def extract_query_terms(query_text):
    query_text = clean_text(query_text)
    raw_terms = query_text.replace(",", " ").split()

    stop_terms = {
        "and", "or", "the", "a", "an", "to", "for",
        "role", "staff", "employee", "karyawan"
    }

    terms = []
    for term in raw_terms:
        if len(term) > 2 and term not in stop_terms:
            terms.append(term)

    return list(dict.fromkeys(terms))


def get_skill_match(query_text, course_skills):
    terms = extract_query_terms(query_text)
    skills_text = clean_text(course_skills)

    matched = [term for term in terms if term in skills_text]

    if not matched:
        return "-"

    return ", ".join(matched[:8])


def build_reason(row):
    skill_match = row.get("skill_match", "-")

    if skill_match != "-":
        return (
            f"Kursus ini direkomendasikan karena memiliki kecocokan dengan skill "
            f"{skill_match}, didukung oleh nilai similarity {row['similarity_score']:.3f} "
            f"serta rating course sebesar {row['rating']}."
        )

    return (
        f"Kursus ini direkomendasikan karena profil kontennya memiliki kemiripan yang baik "
        f"dengan input pengguna, dengan nilai similarity {row['similarity_score']:.3f} "
        f"dan rating {row['rating']}."
    )


def recommend_courses(
    input_skill_text,
    role_text="",
    top_n=5,
    level_filter=None,
    certificate_filter=None,
    organization_filter=None,
    min_rating=0.0,
    min_similarity=0.01
):
    df, vectorizer, tfidf_matrix = load_artifacts()

    query_text = f"{role_text} {input_skill_text}".strip()
    cleaned_query = clean_text(query_text)

    if not cleaned_query:
        return pd.DataFrame()

    query_vector = vectorizer.transform([cleaned_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    result_df = df.copy()
    result_df["similarity_score"] = similarity_scores
    result_df["rating_normalized"] = normalize_rating(result_df["rating"])

    # 80% relevansi konten, 20% kualitas rating
    result_df["final_score"] = (
        0.80 * result_df["similarity_score"] +
        0.20 * result_df["rating_normalized"]
    )

    result_df = result_df[result_df["similarity_score"] >= min_similarity]
    result_df = result_df[result_df["rating"] >= min_rating]

    if not is_all_filter(level_filter):
        result_df = result_df[result_df["difficulty"].isin(level_filter)]

    if not is_all_filter(certificate_filter):
        result_df = result_df[result_df["certificate_type"].isin(certificate_filter)]

    if not is_all_filter(organization_filter):
        result_df = result_df[result_df["organization"].isin(organization_filter)]

    if result_df.empty:
        return result_df

    result_df["skill_match"] = result_df["skills"].apply(
        lambda skills: get_skill_match(query_text, skills)
    )

    result_df = result_df.sort_values(
        by=["final_score", "similarity_score", "rating"],
        ascending=False
    ).head(top_n)

    result_df["recommendation_reason"] = result_df.apply(build_reason, axis=1)

    output_columns = [
        "course_name",
        "organization",
        "difficulty",
        "certificate_type",
        "rating",
        "similarity_score",
        "final_score",
        "skill_match",
        "skills",
        "recommendation_reason"
    ]

    return result_df[output_columns].reset_index(drop=True)


def get_filter_options():
    df, _, _ = load_artifacts()

    return {
        "difficulty": ["All"] + sorted(df["difficulty"].dropna().unique().tolist()),
        "certificate_type": ["All"] + sorted(df["certificate_type"].dropna().unique().tolist()),
        "organization": ["All"] + sorted(df["organization"].dropna().unique().tolist())
    }


if __name__ == "__main__":
    result = recommend_courses(
        input_skill_text="python data analysis machine learning",
        role_text="data analyst",
        top_n=5,
        level_filter=["Beginner"],
        certificate_filter=["All"],
        organization_filter=["All"]
    )

    print(result)