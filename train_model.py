import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from features import preprocess_text, extract_features
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
import logging

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Checkpoint file
CHECKPOINT_FILE = 'models/features_checkpoint.pkl'

def load_checkpoint(file=CHECKPOINT_FILE):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            checkpoint = pickle.load(f)
        logging.info(f"Loaded checkpoint from index {checkpoint['last_index']}")
        return checkpoint['features_list'], checkpoint['last_index']
    return [], -1

def save_checkpoint(features_list, index, file=CHECKPOINT_FILE):
    os.makedirs('models', exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump({'features_list': features_list, 'last_index': index}, f)
    logging.info(f"Saved checkpoint at index {index}")

def process_email(args):
    idx, text = args
    try:
        feats, _, _ = extract_features(text)
        return [feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                feats['url_count'], feats['suspicious_urls'], feats['domain_age_days'],
                feats['google_safe']]
    except:
        return [0, 0.0, 0, 0, 0, np.nan, 0]

# Load dataset
logging.info("Loading dataset...")
try:
    df = pd.read_csv('Phishing_Email.csv')
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit(1)

if df.empty:
    logging.error("Dataset is empty. Check Phishing_Email.csv.")
    exit(1)

# Drop unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    logging.info("Dropped 'Unnamed: 0' column")

# Verify expected columns
expected_columns = ['Email Text', 'Email Type']
if not all(col in df.columns for col in expected_columns):
    logging.error(f"Expected columns {expected_columns}, found {df.columns.tolist()}")
    exit(1)

df['Email Text'] = df['Email Text'].fillna('')
df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
df = df.dropna(subset=['Email Type'])
logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Load checkpoint if exists
features_list, last_index = load_checkpoint()
start_index = last_index + 1 if last_index >= 0 else 0

# Extract features with parallel processing
logging.info("Extracting features with URL checks (may take 10-20 mins)...")
if start_index < len(df['Email Text']):
    batch_size = 500
    for batch_start in range(start_index, len(df['Email Text']), batch_size):
        batch_end = min(batch_start + batch_size, len(df['Email Text']))
        batch = [(i, df['Email Text'].iloc[i]) for i in range(batch_start, batch_end)]
        logging.info(f"Processing batch {batch_start} to {batch_end-1}")
        try:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                batch_features = list(tqdm(pool.imap(process_email, batch), total=len(batch), desc="Processing Batch"))
            features_list.extend(batch_features)
            save_checkpoint(features_list, batch_end - 1)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt detected. Saving checkpoint and exiting...")
            save_checkpoint(features_list, batch_end - 1)
            exit(0)

if not features_list:
    logging.error("No features extracted. Check features.py or dataset.")
    exit(1)

X = pd.DataFrame(features_list, columns=['keyword_count', 'sentiment_neg', 'length',
                                         'url_count', 'suspicious_urls', 'domain_age_days',
                                         'google_safe'])
logging.info(f"Feature matrix shape: {X.shape}")

# Impute NaN values and drop all-NaN columns
logging.info("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
valid_columns = [col for col, values in zip(X.columns, X_imputed.T) if not np.all(np.isnan(values))]
X = pd.DataFrame(X_imputed, columns=X.columns)[valid_columns]
logging.info(f"Feature matrix after imputation: {X.shape}")

# Scale features
logging.info("Scaling features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Enhanced TF-IDF (bigrams, more features)
logging.info("Computing enhanced TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_features = vectorizer.fit_transform(df['Email Text'].apply(preprocess_text)).toarray()
X_tfidf = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
X = pd.concat([X_scaled, X_tfidf], axis=1)
logging.info(f"Final feature matrix shape: {X.shape}")

y = df['Email Type']

# Split data
logging.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train optimized model with tuning
logging.info("Training optimized RandomForest model with hyperparameter tuning...")
model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
logging.info(f"Best parameters: {grid_search.best_params_}")

# Evaluate
logging.info("Evaluating optimized model...")
y_pred = best_model.predict(X_test)
print("Optimized Accuracy:", accuracy_score(y_test, y_pred))
print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# Save optimized model and components to models folder
logging.info("Saving optimized model and components to models folder...")
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/phishing_model_optimized.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Optimized model saved as 'models/phishing_model_optimized.pkl'")