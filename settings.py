import os
from dotenv import load_dotenv
load_dotenv()

SITE_URL = os.environ.get("SITE_URL","https://apps.csipacific.ca")
APP_URL = os.environ.get("APP_URL","http://127.0.0.1:8050")

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

INTERVALS_API_KEY = os.environ.get("INTERVALS_API_KEY")

SPORT_ORG_ENDPOINT = f"/api/registration/organization/"
PROFILE_ENDPOINT = f"/api/registration/profile/"

# Optional (default True): SSL verification
INSIDERS_VERIFY_SSL = True
