import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load TF-IDF matrix & dataframe
with open('../data/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('../data/coursera_df.pkl', 'rb') as f:
    df = pickle.load(f)

# Fungsi rekomendasi
def recommend_courses(input_skills, top_n=5):
    input_text = input_skills.lower()
    input_vec = tfidf.transform([input_text])

    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # Ranking dengan rating
    if 'rating' in df.columns:
        ratings = df['rating'].fillna(df['rating'].mean())
        final_scores = 0.7*sim_scores + 0.3*(ratings/ratings.max())
    else:
        final_scores = sim_scores

    top_indices = np.argsort(final_scores)[::-1][:top_n]
    recommendations = df.iloc[top_indices][['course_name','skills','difficulty','rating','organization','certificate_type']]
    recommendations['similarity_score'] = final_scores[top_indices]
    return recommendations