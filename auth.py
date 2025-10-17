from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret.json',
            SCOPES,
            redirect_uri='http://localhost:8080/oauth2callback'
        )
        creds = flow.run_local_server(port=8080, host='localhost')
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

print("Token generated. Saved to token.json.")