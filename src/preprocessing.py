import re
import pandas as pd

from config import CLEAN_DATA_PATH, ensure_directories, get_raw_data_path


COLUMN_CANDIDATES = {
    "course_name": [
        "course_name",
        "course name",
        "Course Name",
        "course_title",
        "course title",
        "title",
        "name"
    ],
    "skills": [
        "skills",
        "Skills",
        "skill",
        "course_skills",
        "course skills"
    ],
    "difficulty": [
        "difficulty",
        "difficulty level",
        "Difficulty Level",
        "course_difficulty",
        "course difficulty",
        "level",
        "difficulty_level"
    ],
    "rating": [
        "rating",
        "ratings",
        "course rating",
        "Course Rating",
        "course_rating"
    ],
    "organization": [
        "organization",
        "university",
        "University",
        "course_organization",
        "course organization",
        "institution",
        "partner"
    ],
    "certificate_type": [
        "certificate_type",
        "certificate type",
        "course_certificate_type",
        "course certificate type",
        "certificate",
        "type"
    ],
    "course_url": [
        "course url",
        "Course URL",
        "course_url",
        "url",
        "link"
    ],
    "course_description": [
        "course description",
        "Course Description",
        "course_description",
        "description",
        "desc"
    ],
}


def normalize_column_name(column_name: str) -> str:
    """
    Menyamakan format nama kolom agar lebih mudah dicocokkan.
    Contoh:
    'Course Name' -> 'course_name'
    'Difficulty Level' -> 'difficulty_level'
    """
    column_name = str(column_name).strip().lower()
    column_name = column_name.replace("-", "_")
    column_name = column_name.replace(" ", "_")
    column_name = re.sub(r"[^a-z0-9_]", "", column_name)
    return column_name


def read_csv_safely(path):
    """
    Membaca file CSV dengan beberapa kemungkinan encoding.
    Ini berguna karena dataset dari Kaggle kadang tidak selalu UTF-8.
    """
    encodings = ["utf-8", "latin1", "ISO-8859-1"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("File CSV tidak bisa dibaca. Coba cek encoding atau format file.")


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mencocokkan nama kolom asli dataset ke nama kolom standar project.

    Dataset kamu memiliki kolom:
    - Course Name
    - University
    - Difficulty Level
    - Course Rating
    - Course URL
    - Course Description
    - Skills

    Karena dataset tidak punya certificate_type,
    maka certificate_type dibuat otomatis dengan nilai default 'Course'.
    """
    original_columns = list(df.columns)

    normalized_map = {
        normalize_column_name(col): col
        for col in original_columns
    }

    rename_map = {}

    for standard_col, candidates in COLUMN_CANDIDATES.items():
        found_col = None

        for candidate in candidates:
            normalized_candidate = normalize_column_name(candidate)

            if normalized_candidate in normalized_map:
                found_col = normalized_map[normalized_candidate]
                break

        if found_col:
            rename_map[found_col] = standard_col

    df = df.rename(columns=rename_map)

    required_cols = [
        "course_name",
        "skills",
        "difficulty",
        "rating",
        "organization"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing_cols)
            + "\nKolom yang terbaca dari dataset: "
            + ", ".join(original_columns)
        )

    if "certificate_type" not in df.columns:
        df["certificate_type"] = "Course"

    if "course_url" not in df.columns:
        df["course_url"] = ""

    if "course_description" not in df.columns:
        df["course_description"] = ""

    selected_columns = [
        "course_name",
        "skills",
        "difficulty",
        "rating",
        "organization",
        "certificate_type",
        "course_url",
        "course_description"
    ]

    return df[selected_columns].copy()


def clean_text(text):
    """
    Membersihkan teks untuk kebutuhan TF-IDF.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9\s,\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_rating(value):
    """
    Membersihkan nilai rating agar menjadi float.
    Contoh:
    '4.8' -> 4.8
    '4,8' -> 4.8
    """
    if pd.isna(value):
        return 0.0

    value = str(value).strip()
    value = value.replace(",", ".")
    value = re.sub(r"[^0-9.]", "", value)

    try:
        rating = float(value)
    except ValueError:
        rating = 0.0

    if rating < 0:
        rating = 0.0

    return rating


def standardize_difficulty(value):
    """
    Menyamakan kategori difficulty.
    """
    text = clean_text(value)

    if "beginner" in text:
        return "Beginner"

    if "intermediate" in text:
        return "Intermediate"

    if "advanced" in text:
        return "Advanced"

    if "mixed" in text:
        return "Mixed"

    return "Mixed"


def infer_certificate_type(row):
    """
    Karena dataset tidak memiliki kolom certificate_type,
    fungsi ini mencoba menebak tipe certificate dari nama kursus dan deskripsi.
    Jika tidak ditemukan pola khusus, default-nya 'Course'.
    """
    text = (
        str(row.get("course_name", "")) + " " +
        str(row.get("course_description", ""))
    )

    text = clean_text(text)

    if "professional certificate" in text:
        return "Professional Certificate"

    if "specialization" in text:
        return "Specialization"

    if "guided project" in text:
        return "Guided Project"

    if "certificate" in text:
        return "Certificate"

    return "Course"


def standardize_certificate_type(value):
    """
    Menyamakan format certificate_type jika kolomnya memang ada.
    """
    text = clean_text(value)

    if "professional" in text:
        return "Professional Certificate"

    if "specialization" in text:
        return "Specialization"

    if "guided" in text and "project" in text:
        return "Guided Project"

    if "certificate" in text:
        return "Certificate"

    if "course" in text:
        return "Course"

    return "Course"


def prepare_dataset(input_path=None, save_output=True):
    """
    Fungsi utama preprocessing.

    Tahapan:
    1. Load dataset
    2. Mapping nama kolom
    3. Handling missing value
    4. Handling duplicate
    5. Cleaning text
    6. Membuat combined_features untuk TF-IDF
    7. Menyimpan dataset bersih
    """
    ensure_directories()

    if input_path is None:
        input_path = get_raw_data_path()

    df = read_csv_safely(input_path)
    df = map_columns(df)

    df = df.drop_duplicates()
    df = df.drop_duplicates(
        subset=["course_name", "organization"],
        keep="first"
    )

    df["course_name"] = df["course_name"].fillna("Unknown Course").astype(str)
    df["skills"] = df["skills"].fillna("").astype(str)
    df["difficulty"] = df["difficulty"].fillna("Mixed")
    df["rating"] = df["rating"].fillna(0)
    df["organization"] = df["organization"].fillna("Unknown Organization").astype(str)
    df["certificate_type"] = df["certificate_type"].fillna("Course")
    df["course_url"] = df["course_url"].fillna("").astype(str)
    df["course_description"] = df["course_description"].fillna("").astype(str)

    df["difficulty"] = df["difficulty"].apply(standardize_difficulty)
    df["rating"] = df["rating"].apply(clean_rating)

    # Jika semua certificate_type bernilai Course karena kolom aslinya tidak ada,
    # sistem mencoba menebak dari nama kursus dan deskripsi.
    if df["certificate_type"].nunique() == 1 and df["certificate_type"].iloc[0] == "Course":
        df["certificate_type"] = df.apply(infer_certificate_type, axis=1)
    else:
        df["certificate_type"] = df["certificate_type"].apply(standardize_certificate_type)

    df["clean_course_name"] = df["course_name"].apply(clean_text)
    df["clean_skills"] = df["skills"].apply(clean_text)
    df["clean_difficulty"] = df["difficulty"].apply(clean_text)
    df["clean_organization"] = df["organization"].apply(clean_text)
    df["clean_certificate_type"] = df["certificate_type"].apply(clean_text)
    df["clean_course_description"] = df["course_description"].apply(clean_text)

    # Skills dibuat dominan karena fitur utama Content-Based Filtering adalah skills.
    df["combined_features"] = (
        df["clean_skills"] + " " +
        df["clean_skills"] + " " +
        df["clean_skills"] + " " +
        df["clean_course_name"] + " " +
        df["clean_course_description"] + " " +
        df["clean_difficulty"] + " " +
        df["clean_certificate_type"] + " " +
        df["clean_organization"]
    )

    df = df.reset_index(drop=True)

    if save_output:
        df.to_csv(CLEAN_DATA_PATH, index=False)

    return df


if __name__ == "__main__":
    data_path = get_raw_data_path()
    clean_df = prepare_dataset(data_path)

    print("Preprocessing selesai.")
    print(f"Jumlah data bersih: {len(clean_df)}")
    print(f"File disimpan ke: {CLEAN_DATA_PATH}")
    print()
    print("Kolom hasil preprocessing:")
    print(clean_df.columns.tolist())
    print()
    print(clean_df.head())