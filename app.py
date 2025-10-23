import streamlit as st
import pickle

# --- Load cả hai mô hình và vectorizer ---
@st.cache_resource
def load_artifacts():
    model_nb = pickle.load(open("model_naive_bayes.pkl", "rb"))
    model_lr = pickle.load(open("logistic_regression.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model_nb, model_lr, vectorizer

model_nb, model_lr, vectorizer = load_artifacts()

# --- Giao diện chính ---
st.title("📧 Ứng dụng Dự đoán Email Spam")
st.write("Ứng dụng sử dụng **hai mô hình**: Naive Bayes và Logistic Regression để phân loại email.")

# Nhập nội dung email
email_text = st.text_area("✉️ Nhập nội dung email tại đây:", height=200, placeholder="Ví dụ: Congratulations! You have won $1000...")

# Nút dự đoán
if st.button("🔍 Dự đoán"):
    if email_text.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung email.")
    else:
        # Biến đổi văn bản bằng TF-IDF
        features = vectorizer.transform([email_text])

        # --- Naive Bayes ---
        prediction_nb = model_nb.predict(features)[0]
        prob_nb = model_nb.predict_proba(features)[0][1]

        # --- Logistic Regression ---
        prediction_lr = model_lr.predict(features)[0]
        prob_lr = model_lr.predict_proba(features)[0][1]

        # --- Kết quả hiển thị ---
        st.subheader("🧠 Kết quả dự đoán:")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Naive Bayes")
            if prediction_nb == 0:
                st.error(f"🚫 SPAM (xác suất {prob_nb:.2f})")
            else:
                st.success(f"✅ KHÔNG SPAM (xác suất spam {prob_nb:.2f})")

        with col2:
            st.markdown("#### 📈 Logistic Regression")
            if prediction_lr == 0:
                st.error(f"🚫 SPAM (xác suất {prob_lr:.2f})")
            else:
                st.success(f"✅ KHÔNG SPAM (xác suất spam {prob_lr:.2f})")