import os
from dotenv import load_dotenv
load_dotenv()

SITE_URL = os.environ.get("SITE_URL","https://apps.csipacific.ca")
APP_URL = os.environ.get("APP_URL", "https://019d87bb-4317-6fd7-f836-53afee499038.share.connect.posit.cloud")

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")



SPORT_ORG_ENDPOINT = f"/api/registration/organization/"
PROFILE_ENDPOINT = f"/api/registration/profile/"

INSIDERS_USERNAME = "dgeneau@csipacific.ca"
INSIDERS_PASSWORD = os.environ.get("INSIDERS_PASSWORD")
INSIDERS_CLIENT_ID = os.environ.get("INSIDERS_CLIENT_ID")
INSIDERS_CLIENT_SECRET = os.environ.get("INSIDERS_CLIENT_SECRET")
# Optional (default True): SSL verification
INSIDERS_VERIFY_SSL = True
