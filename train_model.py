import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
import joblib
from features import extract_features
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CHECKPOINT_FILE = 'models/xgb/features_checkpoint_dl.pkl'
MODEL_SAVE_PATH = 'models/xgb/phishing_model_lstm.h5'
MAX_WORDS = 10000
MAX_LEN = 200
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

def load_checkpoint(file=CHECKPOINT_FILE):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            checkpoint = pickle.load(f)
        logging.info(f"Loaded checkpoint from index {checkpoint['last_index']}")
        return checkpoint['struct_features'], checkpoint['texts'], checkpoint['last_index']
    return [], [], -1

def save_checkpoint(struct_features, texts, index, file=CHECKPOINT_FILE):
    os.makedirs('models', exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump({'struct_features': struct_features, 'texts': texts, 'last_index': index}, f)
    logging.info(f"Saved checkpoint at index {index}")

def process_row(row):
    try:
        feats, cleaned, urls, combined_text = extract_features(row)
        struct = [feats['keyword_count'], feats['sentiment_neg'], feats['length'],
                  feats['url_count'], feats['suspicious_urls'], feats['domain_age_days'],
                  feats['google_safe']]
        return struct, combined_text
    except Exception as e:
        logging.warning(f"Error processing row: {e}")
        return [0, 0.0, 0, 0, 0, np.nan, 0], ""

# Load dataset
logging.info("Loading dataset...")
try:
    df = pd.read_csv('TREC_07.csv')
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    exit(1)

if df.empty:
    logging.error("Dataset is empty.")
    exit(1)

df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['label'] = df['label'].map({'ham': 0, 'phishing': 1, 'spam': 0})
df = df.dropna(subset=['label'])
logging.info(f"Dataset loaded: {df.shape[0]} rows")

# Load or extract features
struct_list, text_list, last_index = load_checkpoint()
start_index = last_index + 1 if last_index >= 0 else 0

if start_index < len(df):
    batch_size = 500
    for batch_start in range(start_index, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        logging.info(f"Processing batch {batch_start} to {batch_end - 1}")
        try:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(process_row, [row for _, row in batch_df.iterrows()]), total=len(batch_df), desc="Extracting Features"))
            struct_batch, text_batch = zip(*results)
            struct_list.extend(struct_batch)
            text_list.extend(text_batch)
            save_checkpoint(struct_list, text_list, batch_end - 1)
        except KeyboardInterrupt:
            logging.info("Interrupted. Saving checkpoint...")
            save_checkpoint(struct_list, text_list, batch_end - 1)
            exit(0)
        except Exception as e:
            logging.error(f"Batch error: {e}")

if not struct_list:
    logging.error("No features extracted.")
    exit(1)

X_struct = pd.DataFrame(struct_list, columns=['keyword_count', 'sentiment_neg', 'length',
                                              'url_count', 'suspicious_urls', 'domain_age_days',
                                              'google_safe'])
y = df['label'].values

logging.info("Imputing and scaling structured features...")
imputer = SimpleImputer(strategy='mean')
X_struct_imputed = imputer.fit_transform(X_struct)
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct_imputed)

logging.info("Tokenizing text sequences...")
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(text_list)
X_text_seq = tokenizer.texts_to_sequences(text_list)
X_text_padded = pad_sequences(X_text_seq, maxlen=MAX_LEN)

logging.info("Oversampling for class balance...")
ros = RandomOverSampler(random_state=42)
X_struct_res, y_res = ros.fit_resample(X_struct_scaled, y)
X_text_res, _ = ros.fit_resample(X_text_padded, y)

X_struct_train, X_struct_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_struct_res, X_text_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

logging.info("Building LSTM model...")
text_input = Input(shape=(MAX_LEN,), name='text_input')
embedding = Embedding(MAX_WORDS, 128, input_length=MAX_LEN)(text_input)
lstm_out = LSTM(128, dropout=0.5, recurrent_dropout=0.5)(embedding)
lstm_out = BatchNormalization()(lstm_out)

struct_input = Input(shape=(X_struct_train.shape[1],), name='struct_input')
dense_struct = Dense(64, activation='relu')(struct_input)
dense_struct = Dropout(0.5)(dense_struct)

concat = Concatenate()([lstm_out, dense_struct])
dense = Dense(64, activation='relu')(concat)
dense = BatchNormalization()(dense)
dense = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[text_input, struct_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

logging.info(f"Starting training for up to {EPOCHS} epochs...")
history = model.fit(
    [X_text_train, X_struct_train], y_train,
    validation_data=([X_text_test, X_struct_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

loss, acc = model.evaluate([X_text_test, X_struct_test], y_test, verbose=0)
print(f"Final Test Accuracy: {acc:.4f}")
print(f"Training completed after {len(history.history['loss'])} epochs.")

logging.info("Saving tokenizer, imputer, and scaler...")
joblib.dump(tokenizer, 'models/xgb/tokenizer.pkl')
joblib.dump(imputer, 'models/xgb/imputer_dl.pkl')
joblib.dump(scaler, 'models/xgb/scaler_dl.pkl')
print("All components saved to models/xgb/ folder.")