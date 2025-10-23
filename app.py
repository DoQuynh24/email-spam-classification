import streamlit as st
import pickle

# --- Load mô hình và vectorizer ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("logistic_regression.pkl", "rb"))
    vectorizer = pickle.load(open("feature_extraction.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- Giao diện ---
st.title("📧 Dự đoán Email Spam")
st.write("Ứng dụng sử dụng để phân loại email.")

# Nhập nội dung email
email_text = st.text_area("Nhập nội dung email tại đây:", height=200)

# Nút dự đoán
if st.button("Dự đoán"):
    if email_text.strip() == "":
        st.warning("Vui lòng nhập nội dung email.")
    else:
        features = vectorizer.transform([email_text])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"🚫 Kết quả: SPAM (xác suất {probability:.2f})")
        else:
            st.success(f"✅ Kết quả: KHÔNG SPAM (xác suất spam {probability:.2f})")