import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
# ----- Tải dữ liệu -----
@st.cache_data
def load_keywords():
    df = pd.read_csv("keywords_sample.csv")
    df = df.dropna(subset=['keyword'])
    df['keyword'] = df['keyword'].astype(str).str.lower()
    return df
df = load_keywords()
# ----- TF-IDF model -----
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(df['keyword'])
def suggest_keywords(input_text, top_n=5):
    input_vec = vectorizer.transform([input_text])
    keyword_vecs = vectorizer.transform(df['keyword'])
    cosine_sim = cosine_similarity(input_vec, keyword_vecs).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    return df['keyword'].iloc[top_indices]
# ----- Giao diện -----
st.markdown("<h1 style='text-align: center;'>🔍 Gợi ý từ khóa & Liên kết tìm kiếm</h1>", unsafe_allow_html=True)
user_input = st.text_input("Nhập từ khóa bạn muốn tìm:")
# ----- Chọn nơi tìm kiếm -----
search_source = st.selectbox("🌐 Chọn nền tảng để tìm kiếm:", [
    "Google 🔍", "Wikipedia 📚", "Shopee 🛍", "YouTube ▶️"
])
# ----- Từ khóa gợi ý -----
if user_input:
    suggestions = suggest_keywords(user_input)
    st.markdown("### 📌 Từ khóa gợi ý (bấm vào để mở liên kết):")
    for keyword in suggestions:
        encoded_kw = urllib.parse.quote_plus(keyword)
        # Xử lý link theo nền tảng
        if "Google" in search_source:
            link = f"https://www.google.com/search?q={encoded_kw}"
            icon = "🔍"
        elif "Wikipedia" in search_source:
            link = f"https://vi.wikipedia.org/wiki/{encoded_kw}"
            icon = "📚"
        elif "Shopee" in search_source:
            link = f"https://shopee.vn/search?keyword={encoded_kw}"
            icon = "🛍"
        elif "YouTube" in search_source:
            link = f"https://www.youtube.com/results?search_query={encoded_kw}"
            icon = "▶️"
        else:
            link = "#"
            icon = "🔗"
        # Hiển thị keyword có biểu tượng & link
        st.markdown(f"- {icon} [{keyword}]({link})", unsafe_allow_html=True)
