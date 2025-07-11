
# HỆ THỐNG GỢI Ý TỪ KHÓA TÌM KIẾM SỬ DỤNG TF-IDF
# -------------------------
# Đây là notebook hướng dẫn chi tiết từng bước xây dựng hệ thống gợi ý từ khóa sử dụng TF-IDF và Cosine Similarity bằng Python

# 🔹 BƯỚC 1: CÀI ĐẶT VÀ IMPORT THƯ VIỆN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 BƯỚC 2: TẢI DỮ LIỆU
# Em cần tải lên file CSV có cột 'keyword'. Ví dụ: 'keywords.csv'
from google.colab import files
uploaded = files.upload()

# 🔹 BƯỚC 3: ĐỌC DỮ LIỆU
df = pd.read_csv(list(uploaded.keys())[0])
df = df.dropna(subset=['keyword'])  # loại bỏ dòng trống
df['keyword'] = df['keyword'].astype(str).str.lower()  # chuyển về chữ thường
df['keyword'] = df['keyword'].str.replace(r'[^\w\s]', '', regex=True)  # loại bỏ ký tự đặc biệt
df.head()

# 🔹 BƯỚC 4: BIẾN TỪ KHÓA THÀNH VECTOR TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['keyword'])

# 🔹 BƯỚC 5: VIẾT HÀM GỢI Ý
def suggest_keywords(input_text, top_n=5):
    input_vec = vectorizer.transform([input_text])
    cosine_sim = cosine_similarity(input_vec, X).flatten()
    indices = cosine_sim.argsort()[-top_n:][::-1]
    return df['keyword'].iloc[indices]

# 🔹 BƯỚC 6: THỬ NGHIỆM
input_text = "nike shoes"  # Em có thể thay bằng từ khóa khác
suggest_keywords(input_text)
