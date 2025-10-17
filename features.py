import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from math import log
from Levenshtein import distance as lev_distance
import whois
from pysafebrowsing import SafeBrowsing
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import requests
import os
import ssl
import warnings
import pickle
import hashlib

# Suppress all warnings
warnings.filterwarnings("ignore")

# Fix SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data silently
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

PHISH_KEYWORDS = ['urgent', 'account', 'verify', 'login', 'password', 'bank', 'free', 'win', 'click here']
COMMON_DOMAINS = ['google.com', 'paypal.com', 'amazon.com', 'bankofamerica.com']

# Cache for whois and Safe Browsing results
WHOIS_CACHE_FILE = 'models/whois_cache.pkl'
SAFE_BROWSING_CACHE_FILE = 'models/safe_browsing_cache.pkl'

def load_cache(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def save_cache(data, file):
    try:
        os.makedirs('models', exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass

whois_cache = load_cache(WHOIS_CACHE_FILE)
safe_browsing_cache = load_cache(SAFE_BROWSING_CACHE_FILE)

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def extract_features(email_text, api_key=None):
    if not api_key:
        api_key = os.getenv('SAFE_BROWSING_KEY')
    cleaned = preprocess_text(email_text)
    features = {}
    features['keyword_count'] = sum(1 for kw in PHISH_KEYWORDS if kw in cleaned)
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(cleaned)
        features['sentiment_neg'] = sentiment['neg']
    except:
        features['sentiment_neg'] = 0.0
    features['length'] = len(cleaned)
    urls = re.findall(r'http\S+', email_text)
    features['url_count'] = len(urls)
    features['suspicious_urls'] = 0
    features['domain_age_days'] = np.nan
    features['google_safe'] = 0

    for url in urls:
        chars = ''.join(re.findall(r'[a-zA-Z0-9]', url.lower()))
        if len(chars) > 0:
            freq = {c: chars.count(c) / len(chars) for c in set(chars)}
            entropy = -sum(p * log(p) for p in freq.values() if p > 0)
            if entropy > 3.5:
                features['suspicious_urls'] += 1
        domain = re.search(r'://([^/]+)', url).group(1) if re.search(r'://([^/]+)', url) else ''
        if any(lev_distance(domain, common) < 3 for common in COMMON_DOMAINS):
            features['suspicious_urls'] += 1
        # Whois lookup with cache
        domain_hash = hashlib.md5(domain.encode()).hexdigest()
        if domain_hash in whois_cache:
            creation_date = whois_cache[domain_hash]
        else:
            try:
                w = whois.whois(domain)
                creation_date = w.creation_date if hasattr(w, 'creation_date') else None
                whois_cache[domain_hash] = creation_date
                save_cache(whois_cache, WHOIS_CACHE_FILE)
            except:
                creation_date = None
        if creation_date:
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            try:
                age = (pd.Timestamp.now() - pd.to_datetime(creation_date)).days
                features['domain_age_days'] = age if not np.isnan(age) else features['domain_age_days']
                if age < 30:
                    features['suspicious_urls'] += 1
            except:
                pass
        # Redirect check
        try:
            r = requests.head(url, allow_redirects=True, timeout=5)
            if len(r.history) > 2:
                features['suspicious_urls'] += 1
        except:
            pass
        # Safe Browsing with cache
        if api_key:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in safe_browsing_cache:
                features['google_safe'] = safe_browsing_cache[url_hash]
            else:
                try:
                    sb = SafeBrowsing(api_key)
                    result = sb.lookup_urls([url])
                    features['google_safe'] = 1 if result[url]['malicious'] else 0
                    safe_browsing_cache[url_hash] = features['google_safe']
                    save_cache(safe_browsing_cache, SAFE_BROWSING_CACHE_FILE)
                except:
                    pass
    return features, cleaned, urls