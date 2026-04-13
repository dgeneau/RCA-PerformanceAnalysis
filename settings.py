import os
from dotenv import load_dotenv
load_dotenv()

SITE_URL = os.environ.get("SITE_URL","https://apps.csipacific.ca")
APP_URL = os.environ.get("APP_URL","http://127.0.0.1:8050")

AUTH_URL = f"{SITE_URL}/o/authorize"
TOKEN_URL = f"{SITE_URL}/o/token/"
CLIENT_ID = 'bDf3z9KwxSzCFtxabQ10UwlnHCMl2IsE5teZWLu4' #os.environ.get("CLIENT_ID")
CLIENT_SECRET = 'em7L8NeqjKP8vxTEYRz7LrnHKz7aU8pm7t0DfbCiyQkljgz2YEyf7j2wCfWuN3m21QfKehzAwkwBc8boXGYSOJWFm6PAif4iHQ3kbT5xZ5safDeBlt03YDgqr5EhooYR' #os.environ.get("CLIENT_SECRET")

INTERVALS_API_KEY = "69yu8aiqme8lwh0v20vg3lbrm" #os.environ.get("INTERVALS_API_KEY")

SPORT_ORG_ENDPOINT = f"/api/registration/organization/"
PROFILE_ENDPOINT = f"/api/registration/profile/"

INSIDERS_USERNAME = "dgeneau@csipacific.ca"
INSIDERS_PASSWORD = "Joegeneau!1959"
INSIDERS_CLIENT_ID = 'M9d3J3axpMU9z9xMfaqNtOB4BdxmsyPMK2v63yBC'
INSIDERS_CLIENT_SECRET = 'POh9pZ1djjNOtS8rX9FzTYHAv3ARYIvaht9pfXlKPc3axcTaCPoxYehS3OVOGhRUn9ahaujugmftpajWC7zAW0LoxVEBMhhEIg86D4Yp5g05KfT9SLGSrin6oyd0SNnd'

# Optional (default True): SSL verification
INSIDERS_VERIFY_SSL = True



'''
RAW_INGEST_ENDPOINT = f"/api/warehouse/ingestion/primary/"

# UUID-only: pass the DataSource UUID string directly
WELLNESS_SOURCE_UUID = os.environ.get("WELLNESS_SOURCE_UUID")
'''