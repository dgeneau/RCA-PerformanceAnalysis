import os
from dash_auth_external import DashAuthExternal
from settings import AUTH_URL, TOKEN_URL, APP_URL, CLIENT_ID, CLIENT_SECRET

auth = DashAuthExternal(
    AUTH_URL,
    TOKEN_URL,
    app_url=APP_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
)
server = auth.server  # expose the Flask server for app.py