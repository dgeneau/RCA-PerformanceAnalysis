import requests
from settings import SITE_URL

def fetch_options(path, token, label_key, value_key, params=None, limit=1000):
    headers = {"Authorization": f"Bearer {token}"}

    if params and isinstance(params,dict):
        params.update({"limit": limit})
    else:
        params = {"limit": limit}

    resp = requests.get(f"{SITE_URL}{path}", params=params, headers=headers, timeout=5)
    resp.raise_for_status()

    # print("=========================")
    # print(path)

    items = resp.json()

    # print(items)

    if 'results' in items:
        rv = [{"label": item[label_key], "value": item[value_key]} for item in items["results"]]
    elif isinstance(items, list):
        rv = [{"label": val, "value": val} for val in items if val]
    else:
        rv = []

    return rv

def fetch_profile(token, profile_id):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SITE_URL}/api/registration/profile/{profile_id}"
    params = {}  # choose a reasonable chunk size
    all_records = []

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    payload = r.json()

    return payload

def fetch_profiles(token, filters):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{SITE_URL}/api/registration/profile/"
    params = {**filters, "limit": 100, "offset": 0}  # choose a reasonable chunk size
    all_records = []

    while url:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        payload = r.json()
        all_records.extend(payload["results"])

        # Move to the next page
        url = payload.get("next")
        # Once we switch to using `next`, we no longer need `params`
        params = None

    return all_records

def restructure_profile(profile, format='profile'):
    if not format:
        format = 'profile'


    if format == 'profile':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org':None,
            'dob' :profile['person']['dob'] if profile['person'] else None,
            'majority_age': profile['person']['majority_age'] if profile['person'] else None,
            # 'enrollment_status': profile['current_enrollment']['enrollment_status'] if profile[
            #     'current_enrollment'] else None

            'birthplace':f"{profile['birth_city']['name_ascii']}, {profile['birth_city']['province_territory']}" if 'birth_city' in profile and profile['birth_city'] else None,
            'residence': f"{profile['residence_city']['name_ascii']}, {profile['residence_city']['province_territory']}" if 'residence_city' in profile and profile['residence_city'] else None,

            'enrollment_expiry': profile['current_enrollment']['end_date'] if profile[
                'current_enrollment'] else None
        }
    elif format == 'contact':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org': None,
            'dob': profile['person']['dob'] if profile['person'] else None,
            'majority_age': profile['person']['majority_age'] if profile['person'] else None,
            'guardian': f"{profile['person']['guardian']['first_name']} {profile['person']['guardian']['last_name']}" if profile['person']['guardian'] else None,
            'guardian_email': profile['person']['guardian']['email'] if profile['person']['guardian'] else None,
            'emergency_contact': f"{profile['person']['emergency_contact']['first_name']} {profile['person']['emergency_contact']['last_name']} ({profile['person']['emergency_contact']['relationship']})" if profile['person']['emergency_contact'] else None,
            'emergency_contact_phone': profile['person']['emergency_contact']['phone_number'] if profile['person']['emergency_contact'] else None,
        }
    elif format == 'social':
        record = {
            'role': profile['role_slug'] if profile['role_slug'] else None,
            'first_name': profile['person']['first_name'] if profile['person'] else None,
            'last_name': profile['person']['last_name'] if profile['person'] else None,
            'email': profile['person']['email'] if profile['person'] else None,
            'sport': profile['sport']['name'] if profile['sport'] else None,
            'org':None,
        }

        if profile['person']['social_media_accounts']:
            for act in profile['person']['social_media_accounts']:
                record[act['platform']] = act['username']

    if profile['role_slug'] == 'staff':
        record['org'] = profile['organization']['name'] if profile['organization'] else None
    else:
        record['org'] = profile['current_nomination']['organization']['name'] if profile['current_nomination'] else None

    return record