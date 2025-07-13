import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ------------------- 1. Đọc dữ liệu từ khóa -------------------
df = pd.read_csv("keywords_sample.csv")
keywords = df['keyword'].astype(str).tolist()
# ------------------- 2. Hàm lưu lịch sử -------------------
def save_query(user_id, query):
    try:
        history_df = pd.read_csv("search_history.csv")
    except:
        history_df = pd.DataFrame(columns=["user_id", "query"])
    new_row = {"user_id": user_id, "query": query}
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv("search_history.csv", index=False)

# ------------------- 3. Lấy lịch sử người dùng -------------------
def get_user_history(user_id):
    try:
        history_df = pd.read_csv("search_history.csv")
        return history_df[history_df["user_id"] == user_id]["query"].tolist()
    except:
        return []
# ------------------- 4. Hàm gợi ý từ khóa -------------------
def suggest_keywords(user_input, user_id, top_n=5):
    user_history = get_user_history(user_id)
    # Tạo tập văn bản mở rộng gồm: user_input + lịch sử + từ khóa
    documents = [user_input] + user_history + keywords
    # TF-IDF và cosine
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    # Bỏ phần lịch sử đầu tiên, chỉ lấy phần tương đồng với các từ khóa
    sim_scores = cosine_sim[0][-len(keywords):]
    # Trả về top N từ khóa liên quan
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return [keywords[i] for i in top_indices]
# ------------------- 5. Giao diện Streamlit -------------------
st.title("🔍 Gợi ý từ khóa thông minh")
user_id = "user1"  # Tạm thời giả định
user_input = st.text_input("Nhập từ khóa bạn đang tìm kiếm:")
if user_input:
    save_query(user_id, user_input)
    suggestions = suggest_keywords(user_input, user_id)
    st.subheader("🔎 Từ khóa được đề xuất:")
    for i, sug in enumerate(suggestions, 1):
        st.write(f"{i}. {sug}")
