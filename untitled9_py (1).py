import streamlit as st
import pandas as pd
import os
import glob
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Cài đặt giao diện
st.set_page_config(page_title="Keyword Recommender", layout="wide")
st.markdown("<h1 style='text-align:center;'>✨ Gợi Ý Từ Khóa Cá Nhân Hóa</h1>", unsafe_allow_html=True)
# Tải từ khóa
@st.cache_data
def load_keywords():
    df = pd.read_csv("keywords_sample.csv")
    df = df.dropna(subset=['keyword'])
    df['keyword'] = df['keyword'].astype(str).str.lower()
    return df
df = load_keywords()
# Lấy danh sách người dùng
def get_all_users():
    files = glob.glob("search_history_*.csv")
    return sorted([os.path.splitext(os.path.basename(f))[0].replace("search_history_", "") for f in files])
# Chọn người dùng
col1, col2 = st.columns([2, 3])
with col1:
    new_user = st.text_input("🆕 Nhập tên người dùng mới:")
with col2:
    existing_users = get_all_users()
    selected_user = st.selectbox("📂 Hoặc chọn người dùng đã có:", existing_users if existing_users else [""])
if new_user:
    current_user = new_user.strip().lower()
else:
    current_user = selected_user.strip().lower()
if current_user:
    history_file = f"search_history_{current_user}.csv"
# Load lịch sử người dùng
    def load_user_history(file_path):
        try:
            history_df = pd.read_csv(file_path)
            return " ".join(history_df['keyword'].astype(str).tolist())
        except:
            return ""
    user_history = load_user_history(history_file)
 # TF-IDF MODEL 
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(df['keyword'])
    def suggest_keywords(input_text, top_n=5):
        personalized_input = input_text + " " + user_history
        input_vec = vectorizer.transform([personalized_input])
        keyword_vecs = vectorizer.transform(df['keyword'])
        cosine_sim = cosine_similarity(input_vec, keyword_vecs).flatten()
        top_indices = cosine_sim.argsort()[::-1][:top_n]
        return df['keyword'].iloc[top_indices]
# Chọn nền tảng tìm kiếm
    st.markdown("### 🌐 Chọn nền tảng để tìm kiếm:")
    search_source = st.radio(
        "Nền tảng:",
        ["Google 🔍", "Wikipedia 📚", "Shopee 🛍", "YouTube ▶️"],
        horizontal=True,
    )
    # Nhập từ khóa
    user_input = st.text_input("✏️ Nhập từ khóa bạn muốn tìm:")
    if user_input:
        suggestions = suggest_keywords(user_input)
        st.markdown("### 📌 Gợi ý từ khóa cho bạn:")
        cols = st.columns(2)
        for i, keyword in enumerate(suggestions):
            encoded_kw = urllib.parse.quote_plus(keyword)
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
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div style='background-color:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:10px;border-left:5px solid #4CAF50'>
                        <strong>{icon} <a href="{link}" target="_blank" style="text-decoration:none;color:#333">{keyword.title()}</a></strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
        # Lưu lịch sử
        new_entry = pd.DataFrame({'keyword': [user_input]})
        try:
            old_df = pd.read_csv(history_file)
            updated_df = pd.concat([old_df, new_entry], ignore_index=True)
        except:
            updated_df = new_entry
        updated_df.to_csv(history_file, index=False)
    # Gợi ý ngẫu nhiên mỗi lần truy cập
    st.markdown("---")
    st.markdown("### 🎁 Gợi ý từ khóa hôm nay:")
    sample_kw = df.sample(3)['keyword'].tolist()
    for kw in sample_kw:
        st.markdown(f"- 🌟 **{kw.title()}**")
    # Hiển thị lịch sử cũ
    st.markdown("---")
    if os.path.exists(history_file):
        hist_df = pd.read_csv(history_file)
        if not hist_df.empty:
            st.markdown("### 📜 Lịch sử tìm kiếm gần đây:")
            st.write(hist_df.tail(5)['keyword'].tolist())
