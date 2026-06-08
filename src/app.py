import streamlit as st
from model import recommend_courses

st.title("Smart Skill Gap Resolver 🚀")
st.markdown("Masukkan skill yang ingin dikembangkan, sistem akan merekomendasikan kursus relevan.")

user_skill = st.text_input("Skill / Kompetensi (pisahkan koma jika lebih dari satu)", "")
top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if user_skill.strip() != "":
        results = recommend_courses(user_skill, top_n)
        st.dataframe(results)
    else:
        st.warning("Silakan masukkan skill terlebih dahulu!")