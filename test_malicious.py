from features import extract_features

sample_email = """
Subject: Urgent: Your Account is Compromised!
Click here to secure: http://testsafebrowsing.appspot.com/s/malware.html
"""
features, cleaned, urls = extract_features(sample_email)
print("Features:", features)
print("Cleaned Text:", cleaned)
print("URLs:", urls)