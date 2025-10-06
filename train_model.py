import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from features import preprocess_text, extract_features
from tqdm import tqdm
import pickle
import os

# Checkpoint file
CHECKPOINT_FILE = 'features_checkpoint.pkl'

def load_checkpoint(file=CHECKPOINT_FILE):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['features_list'], checkpoint['last_index']
    return [], -1

def save_checkpoint(features_list, index, file=CHECKPOINT_FILE):
    with open(file, 'wb') as f:
        pickle.dump({'features_list': features_list, 'last_index': index}, f)

# Load dataset
try:
    df = pd.read_csv('Phishing_Email.csv')
except:
    exit(1)

if df.empty:
    exit(1)

# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Verify expected columns
expected_columns = ['Email Text', 'Email Type']
if not all(col in df.columns for col in expected_columns):
    exit(1)

df['Email Text'] = df['Email Text'].fillna('')
df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
df = df.dropna(subset=['Email Type'])

# Load checkpoint if exists
features_list, last_index = load_checkpoint()
start_index = last_index + 1 if last_index >= 0 else 0

# Extract features with URL checks
if start_index < len(df['Email Text']):
    for idx, text in tqdm(enumerate(df['Email Text'].iloc[start_index:]), total=len(df['Email Text']) - start_index, 
                         desc="Processing Emails", initial=start_index):
        try:
            feats, _, _ = extract_features(text)
            features_list.append([feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                                 feats['url_count'], feats['suspicious_urls'], feats['domain_age_days'],
                                 feats['google_safe']])
            # Save checkpoint every 100 emails
            if (idx + start_index + 1) % 100 == 0:
                save_checkpoint(features_list, idx + start_index)
        except:
            features_list.append([0, 0.0, 0, 0, 0, np.nan, 0])

    # Save final checkpoint
    save_checkpoint(features_list, len(df['Email Text']) - 1)

if not features_list:
    exit(1)

X = pd.DataFrame(features_list, columns=['keyword_count', 'sentiment_neg', 'length',
                                         'url_count', 'suspicious_urls', 'domain_age_days',
                                         'google_safe'])

# Impute NaN values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# TF-IDF for text
vectorizer = TfidfVectorizer(max_features=100)
tfidf_features = vectorizer.fit_transform(df['Email Text'].apply(preprocess_text)).toarray()
X_tfidf = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
X = pd.concat([X_scaled, X_tfidf], axis=1)

y = df['Email Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# Save model and vectorizer
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model, vectorizer, imputer, and scaler saved as 'phishing_model.pkl', 'tfidf_vectorizer.pkl', 'imputer.pkl', and 'scaler.pkl'")