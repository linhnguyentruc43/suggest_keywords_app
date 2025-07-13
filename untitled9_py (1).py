import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
# ----- 1. Tải dữ liệu chính -----
@st.cache_data
def load_data():
    df = pd.read_csv("keywords_sample.csv")
    df = df.dropna(subset=['keyword'])
    df['keyword'] = df['keyword'].astype(str).str.lower()
    return df
df = load_data()
# ----- 2. Tìm tất cả người dùng từ các file lịch sử -----
def get_all_users():
    files = glob.glob("search_history_*.csv")
    users = [os.path.splitext(os.path.basename(f))[0].replace("search_history_", "") for f in files]
    return sorted(users)
# ----- 3. Giao diện chọn người dùng -----
st.markdown("<h1 style='text-align: center;'>🔍 Gợi ý từ khóa cá nhân hóa</h1>", unsafe_allow_html=True)
all_users = get_all_users()
new_user_input = st.text_input("🆕 Nhập tên người dùng mới nếu chưa có:")
selected_user = None
if new_user_input:
    selected_user = new_user_input.strip().lower()
else:
    if all_users:
        selected_user = st.selectbox("👤 Chọn người dùng đã có:", all_users)
    else:
        st.info("🔧 Chưa có người dùng nào. Vui lòng nhập tên mới.")
if selected_user:
    history_file = f"search_history_{selected_user}.csv"
    def load_user_history(file_path):
        try:
            history_df = pd.read_csv(file_path)
            return " ".join(history_df['keyword'].astype(str).tolist())
        except:
            return ""
    user_history = load_user_history(history_file)
    # ----- TF-IDF model -----
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(df['keyword'])
    def suggest_keywords(input_text, top_n=5):
        personalized_input = input_text + " " + user_history
        input_vec = vectorizer.transform([personalized_input])
        keyword_vecs = vectorizer.transform(df['keyword'])
        cosine_sim = cosine_similarity(input_vec, keyword_vecs).flatten()
        top_indices = cosine_sim.argsort()[::-1][:top_n]
        return df['keyword'].iloc[top_indices]
    # ----- Giao diện tìm kiếm -----
    user_input = st.text_input("🔎 Nhập từ khóa bạn đang tìm:")
    if user_input:
        suggestions = suggest_keywords(user_input)
        st.markdown("### 📌 Gợi ý từ khóa dành riêng cho bạn:")
        for i, keyword in enumerate(suggestions, 1):
            st.write(f"{i}. {keyword}")
        # ----- Lưu lịch sử tìm kiếm -----
        new_entry = pd.DataFrame({'keyword': [user_input]})
        try:
            history_df = pd.read_csv(history_file)
            updated_df = pd.concat([history_df, new_entry], ignore_index=True)
        except:
            updated_df = new_entry
        updated_df.to_csv(history_file, index=False)
