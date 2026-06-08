import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv('../data/Coursera.csv')

# Bersihkan kolom skills
df['skills_clean'] = df['skills'].fillna('').str.lower().str.replace(r'[^\w, ]','', regex=True)
df['text_features'] = df['skills_clean'] + ' ' + df['difficulty'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Simpan TF-IDF matrix & dataframe
with open('../data/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open('../data/coursera_df.pkl', 'wb') as f:
    pickle.dump(df, f)

print("Preprocessing selesai, TF-IDF dan dataset disimpan.")