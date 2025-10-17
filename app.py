import streamlit as st
import pandas as pd
import numpy as np
import joblib
from features import preprocess_text, extract_features

st.set_page_config(
    page_title="Phishing Email Detector",  
    page_icon=" 🤿"   
)
# Load model and components from models folder
model = joblib.load('models/phishing_model_optimized.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
imputer = joblib.load('models/imputer.pkl')
scaler = joblib.load('models/scaler.pkl')

# Streamlit UI
st.title("🛡️ AI-Powered Phishing Detection")
st.write("Scan emails or URLs in seconds. Powered by 94.80% accurate RandomForest model.")

# Sidebar for tips
with st.sidebar:
    st.header("Quick Tips")
    st.write("=> Look for urgent language or suspicious links.")
    st.write("=> Always hover over URLs before clicking .")
    st.write("=> Report phishing to chronicle613 @gmail.com.")

# Input form
st.header("Scan Input")
user_input = st.text_area("Paste Email Text or URL", height=200, placeholder="e.g., Subject: Urgent! Verify at http://example.com/login")

if st.button("🔍 Scan for Phishing", type="primary"):
    if user_input.strip():
        with st.spinner("Scanning..."):
            # Extract features
            feats, cleaned_text, urls = extract_features(user_input)
            features = [feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                        feats['url_count'], feats['suspicious_urls'], feats['domain_age_days'],
                        feats['google_safe']]
            
            # Prepare feature DataFrame
            feature_df = pd.DataFrame([features], columns=['keyword_count', 'sentiment_neg', 'length',
                                                          'url_count', 'suspicious_urls', 'domain_age_days',
                                                          'google_safe'])
            
            # Impute and scale
            feature_df = pd.DataFrame(imputer.transform(feature_df), columns=feature_df.columns)
            feature_df = pd.DataFrame(scaler.transform(feature_df), columns=feature_df.columns)
            
            # TF-IDF
            tfidf_features = vectorizer.transform([preprocess_text(user_input)]).toarray()
            tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            
            # Combine features
            final_features = pd.concat([feature_df, tfidf_df], axis=1)
            
            # Predict
            prediction = model.predict(final_features)[0]
            probability = model.predict_proba(final_features)[0][1]
            
            # Display result
            col1, col2 = st.columns(2)
            if prediction == 1:
                with col1:
                    st.error(f"⚠️ Phishing Detected!")
                with col2:
                    st.metric("Confidence", f"{probability:.1%}")
                st.warning("Tip: Do not click links. Report to security team.")
            else:
                with col1:
                    st.success(f"✅ Safe!")
                with col2:
                    st.metric("Confidence", f"{1 - probability:.1%}")
                st.info("Tip: Still verify sender and links manually.")
            
            # Progress bar for confidence
            st.progress(probability)
            
            # Show features
            st.subheader("📊 Extracted Features")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Keywords", feats['keyword_count'])
                st.metric("Negative Sentiment", f"{feats['sentiment_neg']:.2f}")
            with col2:
                st.metric("Length (chars)", feats['length'])
                st.metric("URLs Found", feats['url_count'])
            with col3:
                st.metric("Suspicious URLs", feats['suspicious_urls'])
                st.metric("Domain Age", f"{feats['domain_age_days']:.0f} days" if not np.isnan(feats['domain_age_days']) else "N/A")
                st.metric("Safe Browsing", "Malicious" if feats['google_safe'] == 1 else "Safe")
            
            # Cleaned text
            st.subheader("🔍 Cleaned Text Analysis")
            st.write(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
            
            # URLs
            if urls:
                st.subheader("🔗 URLs Found")
                for url in urls:
                    st.code(url)
    else:
        st.warning("Please enter an email or URL to scan.")

# Portfolio note
st.markdown("---")
st.markdown("Built by Sumukha Vasista for cybersecurity portfolio (2026–27). Model accuracy: 94.80%.")