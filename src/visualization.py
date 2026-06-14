import pandas as pd
import plotly.express as px

from preprocessing import clean_text


def get_summary_metrics(df):
    total_courses = len(df)
    total_organizations = df["organization"].nunique()
    avg_rating = round(df["rating"].mean(), 2)
    total_certificate_types = df["certificate_type"].nunique()

    return {
        "total_courses": total_courses,
        "total_organizations": total_organizations,
        "avg_rating": avg_rating,
        "total_certificate_types": total_certificate_types
    }


def plot_difficulty_distribution(df):
    data = df["difficulty"].value_counts().reset_index()
    data.columns = ["difficulty", "count"]

    fig = px.bar(
        data,
        x="difficulty",
        y="count",
        title="Distribusi Level Kesulitan Kursus",
        text="count"
    )

    fig.update_layout(
        xaxis_title="Difficulty",
        yaxis_title="Jumlah Kursus",
        title_x=0.02
    )

    return fig


def plot_certificate_distribution(df):
    data = df["certificate_type"].value_counts().reset_index()
    data.columns = ["certificate_type", "count"]

    fig = px.pie(
        data,
        names="certificate_type",
        values="count",
        title="Distribusi Tipe Sertifikat"
    )

    fig.update_layout(title_x=0.02)

    return fig


def plot_top_organizations(df, top_n=10):
    data = df["organization"].value_counts().head(top_n).reset_index()
    data.columns = ["organization", "count"]

    fig = px.bar(
        data,
        x="count",
        y="organization",
        orientation="h",
        title=f"Top {top_n} Penyedia Kursus",
        text="count"
    )

    fig.update_layout(
        xaxis_title="Jumlah Kursus",
        yaxis_title="Organization",
        title_x=0.02,
        yaxis={"categoryorder": "total ascending"}
    )

    return fig


def plot_rating_distribution(df):
    fig = px.histogram(
        df,
        x="rating",
        nbins=20,
        title="Distribusi Rating Kursus"
    )

    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Jumlah Kursus",
        title_x=0.02
    )

    return fig


def get_top_skills(df, top_n=20):
    skill_items = []

    for skills in df["skills"].fillna(""):
        skills = str(skills)

        if "," in skills:
            parts = skills.split(",")
        else:
            parts = skills.split()

        for skill in parts:
            cleaned = clean_text(skill).strip()

            if len(cleaned) > 2:
                skill_items.append(cleaned)

    skill_series = pd.Series(skill_items)

    if skill_series.empty:
        return pd.DataFrame(columns=["skill", "count"])

    top_skills = skill_series.value_counts().head(top_n).reset_index()
    top_skills.columns = ["skill", "count"]

    return top_skills


def plot_top_skills(df, top_n=20):
    data = get_top_skills(df, top_n)

    fig = px.bar(
        data,
        x="count",
        y="skill",
        orientation="h",
        title=f"Top {top_n} Skill yang Sering Muncul",
        text="count"
    )

    fig.update_layout(
        xaxis_title="Frekuensi",
        yaxis_title="Skill",
        title_x=0.02,
        yaxis={"categoryorder": "total ascending"}
    )

    return fig