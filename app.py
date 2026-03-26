import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved model and vectorizer (you should save them beforehand)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Fake Job Detector", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}
.stButton button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚨 AI Fake Job Detector")
st.caption("Analyze job descriptions and detect fraudulent postings using NLP + Explainable AI")

# Layout
col1, col2 = st.columns([2,1])

with col1:
    user_input = st.text_area("📄 Paste Job Description", height=250, placeholder="Paste job description here...")

with col2:
    st.markdown("### 🧠 What this app does")
    st.write("- Detects fake job postings")
    st.write("- Shows confidence score")
    st.write("- Highlights suspicious words")

# Predict button
if st.button("🔍 Analyze Job"):
    if user_input.strip() == "":
        st.warning("Please enter a job description")
    else:
        # Transform input
        input_vec = vectorizer.transform([user_input])
        
        # Prediction
        pred = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0][1]
        
        # Get feature names and weights
        feature_names = vectorizer.get_feature_names_out()
        weights = model.coef_[0]
        
        # Contribution
        input_array = input_vec.toarray()[0]
        contributions = input_array * weights
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions
        })
        
        # Separate fake and real indicators
        fake_indicators = feature_importance[feature_importance['contribution'] > 0].sort_values(by='contribution', ascending=False)
        real_indicators = feature_importance[feature_importance['contribution'] < 0].sort_values(by='contribution', ascending=True)
        
        # Result UI
        st.markdown("---")
        
        if pred == 1:
            st.error(f"⚠️ FAKE JOB DETECTED ({prob*100:.2f}% confidence)")
        else:
            st.success(f"✅ REAL JOB ({(1-prob)*100:.2f}% confidence)")
        
        st.markdown("### 🔍 Top Influential Words")
        
        # Show fake indicators
        if not fake_indicators.empty:
            st.markdown("#### 🚨 FAKE INDICATORS (pushing towards FAKE)")
            for _, row in fake_indicators.head(5).iterrows():
                st.write(f"• {row['feature']} ({row['contribution']:.3f})")
        
        # Show real indicators  
        if not real_indicators.empty:
            st.markdown("#### ✅ REAL INDICATORS (pushing towards REAL)")
            for _, row in real_indicators.head(5).iterrows():
                st.write(f"• {row['feature']} ({row['contribution']:.3f})")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using NLP, TF-IDF, and Logistic Regression")
