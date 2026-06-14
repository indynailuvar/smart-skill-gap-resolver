import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COURSE_DF_PATH,
    ensure_directories,
    get_raw_data_path
)
from preprocessing import prepare_dataset


def train_model():
    ensure_directories()

    raw_data_path = get_raw_data_path()
    df = prepare_dataset(raw_data_path, save_output=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    joblib.dump(df, COURSE_DF_PATH)

    print("Training model selesai.")
    print(f"Total kursus: {len(df)}")
    print(f"Jumlah fitur TF-IDF: {len(vectorizer.get_feature_names_out())}")
    print(f"Vectorizer disimpan ke: {TFIDF_VECTORIZER_PATH}")
    print(f"Matrix disimpan ke: {TFIDF_MATRIX_PATH}")
    print(f"DataFrame disimpan ke: {COURSE_DF_PATH}")


if __name__ == "__main__":
    train_model()