# app.py - Triển khai trên Streamlit
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
# Tiêu đề giao diện
st.title("🔍 Ứng dụng gợi ý từ khóa tìm kiếm")
# BƯỚC 1: TẢI DỮ LIỆU
@st.cache_data
def load_data():
    df = pd.read_csv("keywords_sample.csv")
    df = df.dropna(subset=['keyword'])
    df['keyword'] = df['keyword'].astype(str).str.lower()
    df['keyword'] = df['keyword'].str.replace(r"[^\w\s]", "", regex=True)
    return df
df = load_data()
# BƯỚC 2: CHUYỂN TỪ KHÓA SANG VECTƠ TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['keyword'])
# BƯỚC 3: NHẬP TỪ KHÓA VÀ GỢI Ý
input_text = st.text_input("Nhập từ khóa bạn muốn tìm:")
def suggest_keywords(input_text, top_n=5):
    input_vec = vectorizer.transform([input_text])
    cosine_sim = cosine_similarity(input_vec, X).flatten()
    indices = cosine_sim.argsort()[-top_n:][::-1]
    return df['keyword'].iloc[indices]
if input_text:
    st.subheader("🔎 Các từ khóa gợi ý:")
    suggestions = suggest_keywords(input_text)
    for i, kw in enumerate(suggestions, 1):
        st.write(f"{i}. {kw}")
