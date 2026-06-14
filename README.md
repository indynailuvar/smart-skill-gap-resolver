# smart-skill-gap-resolver
# Smart Skill Gap Resolver

Smart Skill Gap Resolver adalah sistem rekomendasi kursus untuk karyawan menggunakan pendekatan Content-Based Filtering dengan TF-IDF dan Cosine Similarity.

## Business Problem

Perusahaan membutuhkan sistem yang dapat membantu HR merekomendasikan kursus pelatihan berdasarkan skill gap karyawan. Dengan banyaknya pilihan kursus online, proses pemilihan kursus secara manual menjadi kurang efisien.

## Dataset

Dataset yang digunakan adalah Coursera Courses Dataset 2021.

Kolom utama yang digunakan:

- course_name
- skills
- difficulty
- rating
- organization
- certificate_type

## Method

Tahapan pengerjaan:

1. Data Understanding
2. Data Preprocessing
3. Feature Engineering
4. TF-IDF Vectorization
5. Cosine Similarity
6. Filtering
7. Rating-Based Ranking
8. Dashboard Visualization

## Formula Ranking

```python
final_score = 0.80 * similarity_score + 0.20 * rating_normalized