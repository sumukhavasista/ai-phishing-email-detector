from flask import Flask, request, jsonify, redirect, url_for
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request  # Explicit import
import joblib
from features import preprocess_text, extract_features
import pandas as pd
import numpy as np
import base64
import os

app = Flask(__name__)
model = joblib.load('models/phishing_model_optimized.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
imputer = joblib.load('models/imputer.pkl')
scaler = joblib.load('models/scaler.pkl')

# Gmail API setup 
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)

# OAuth callback
@app.route('/oauth2callback')
def oauth2callback():
    flow = Flow.from_client_secrets_file('client_secret.json', SCOPES, redirect_uri='http://localhost:8080/oauth2callback')
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    return redirect(url_for('predict'))

def get_gmail_service():
    global creds
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            return None
    return build('gmail', 'v1', credentials=creds)

def get_email_content(message_id):
    try:
        service = get_gmail_service()
        if not service:
            return "Error: Authentication required"
        message = service.users().messages().get(userId='me', id=message_id).execute()
        payload = message['payload']
        headers = payload.get('headers', [])
        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), '')
        parts = payload.get('parts', [])
        body = ''
        if parts:
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        else:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        return f"Subject: {subject}\n{body}"
    except Exception as e:
        return f"Error: {str(e)}"

def label_email(message_id, prediction):
    try:
        service = get_gmail_service()
        if not service:
            return
        label_name = 'Phishing' if prediction == 1 else 'Safe'
        label = {'name': label_name}
        labels = service.users().labels().list(userId='me').execute()
        label_id = None
        for l in labels['labels']:
            if l['name'] == label_name:
                label_id = l['id']
                break
        if not label_id:
            created_label = service.users().labels().create(userId='me', body=label).execute()
            label_id = created_label['id']
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'addLabelIds': [label_id]}
        ).execute()
    except Exception as e:
        print(f"Error labeling email: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message_id = data.get('message_id')
        if message_id:
            text = get_email_content(message_id)
            if text.startswith("Error"):
                return jsonify({'error': text}), 401
        else:
            text = data['text']
        feats, cleaned_text, urls = extract_features(text)
        features = [feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                    feats['url_count'], feats['suspicious_urls'], feats['domain_age_days'],
                    feats['google_safe']]
        feature_df = pd.DataFrame([features], columns=['keyword_count', 'sentiment_neg', 'length',
                                                      'url_count', 'suspicious_urls', 'domain_age_days',
                                                      'google_safe'])
        feature_df = pd.DataFrame(imputer.transform(feature_df), columns=feature_df.columns)
        feature_df = pd.DataFrame(scaler.transform(feature_df), columns=feature_df.columns)
        tfidf_features = vectorizer.transform([preprocess_text(text)]).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        final_features = pd.concat([feature_df, tfidf_df], axis=1)
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0][1] * 100
        if message_id and prediction == 1:
            label_email(message_id, prediction)
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probability),
            'cleaned_text': cleaned_text[:500],
            'urls': urls
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8080)