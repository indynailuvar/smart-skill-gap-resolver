from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "model"

RAW_DATA_PATH = RAW_DIR / "Coursera.csv"
LEGACY_RAW_DATA_PATH = DATA_DIR / "Coursera.csv"

CLEAN_DATA_PATH = PROCESSED_DIR / "coursera_clean.csv"
TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.pkl"
COURSE_DF_PATH = MODEL_DIR / "coursera_df.pkl"


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_data_path():
    """
    Supaya project tetap jalan meskipun file Coursera.csv
    masih berada langsung di folder data/.
    """
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH

    if LEGACY_RAW_DATA_PATH.exists():
        return LEGACY_RAW_DATA_PATH

    return RAW_DATA_PATH