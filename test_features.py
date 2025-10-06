from features import extract_features

sample_email = """
Subject: Urgent Account Verification
Dear User, click here: http://fakebank.com/login to verify your account.
"""
features, cleaned, urls = extract_features(sample_email)
print("Features:", features)
print("Cleaned Text:", cleaned)
print("URLs:", urls)