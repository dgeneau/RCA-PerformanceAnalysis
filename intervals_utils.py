# intervals_utils.py
'''
Pulling from the 'intervals.icu' API

'''






# at top
import io
import json
import re
from typing import List, Optional, Literal, Dict, Any, Union, Iterable
import pandas as pd
import requests
from datetime import date
from dateutil.parser import isoparse
import base64
import numpy as np



ISO_LOCAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

class IntervalsICUClient:
    BASE = "https://intervals.icu/api/v1"

    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None, timeout: int = 60):
        if not api_key and not access_token:
            raise ValueError("Provide either api_key or access_token.")
        self.api_key = api_key
        self.access_token = access_token
        self.timeout = timeout
        self.session = requests.Session()
        if self.access_token:
            self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})
        self.session.headers.update({"Accept": "application/json"})

    def _auth(self) -> Optional[tuple]:
        return ("API_KEY", self.api_key) if self.api_key and not self.access_token else None

    @staticmethod
    def _fmt_date(d: Union[str, date, None]) -> Optional[str]:
        if d is None:
            return None
        if isinstance(d, date):
            return d.strftime("%Y-%m-%d")
        return isoparse(d).date().strftime("%Y-%m-%d")

    # ---------- helper to parse either NDJSON or JSON array ----------
    @staticmethod
    def _parse_activities_payload(text: str) -> List[Dict[str, Any]]:
        s = text.lstrip("\ufeff").strip()
        if not s:
            return []
        # If it's a JSON array, just load it
        if s.startswith("["):
            data = json.loads(s)
            if isinstance(data, list):
                return data
            raise ValueError("Expected a JSON array from activities endpoint.")
        # Otherwise, assume NDJSON (one JSON object per line)
        rows: List[Dict[str, Any]] = []
        for raw in s.splitlines():
            line = raw.strip()
            if not line or line in ("[", "]", ","):
                continue
            if line.endswith(","):
                line = line[:-1].rstrip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def get_activities(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Optional[Union[str, date]] = None,
        newest: Optional[Union[str, date]] = None,
        limit: Optional[int] = None,
        format: Literal["json", "csv"] = "json",
    ) -> pd.DataFrame:

        params: Dict[str, Any] = {}
        if oldest:
            params["oldest"] = self._fmt_date(oldest)
        if newest:
            params["newest"] = self._fmt_date(newest)
        if limit:
            params["limit"] = int(limit)

        if format == "csv":
            url = f"{self.BASE}/athlete/{athlete_id}/activities.csv"
            r = self.session.get(url, params=params, timeout=self.timeout, auth=self._auth())
            r.raise_for_status()
            # robust to encoding quirks
            return pd.read_csv(io.BytesIO(r.content))

        # JSON path (accept NDJSON or JSON array)
        url = f"{self.BASE}/athlete/{athlete_id}/activities"
        r = self.session.get(url, params=params, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()

        rows = self._parse_activities_payload(r.text)
        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows)
        for col in ("start_date_local", "start_date", "created", "updated"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def get_activity(self, activity_id: Union[int, str]) -> pd.DataFrame:
            """
            Fetch a single activity by its ID using GET /api/v1/activity/{id}.
            Returns: DataFrame (1 row) with all activity details.
            """
            url = f"{self.BASE}/activity/{activity_id}"
            r = self.session.get(url, timeout=self.timeout, auth=self._auth())
            r.raise_for_status()

            data = r.json()
            df = pd.json_normalize(data)

            # Convert date/time columns where applicable
            for col in ("start_date_local", "start_date", "created", "updated"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            return df

    def get_activity_streams(
            self,
            activity_id: Union[int, str],
            types: Optional[Iterable[str]] = None,
            format: str = "csv",                 # "csv" (recommended) or "json"
        ) -> pd.DataFrame:
            """
            Download activity time-series streams as a pandas DataFrame.

            Args:
                activity_id: Intervals activity ID.
                types: Iterable of stream names, e.g. ["time", "watts", "heartrate"].
                    If None, server default is used (often returns common streams).
                format: "csv" (robust) or "json".

            Returns:
                DataFrame with one column per stream (CSV) or normalized JSON.
            """
            q = {}
            if types:
                q["types"] = ",".join(types)

            if format.lower() == "csv":
                url = f"{self.BASE}/activity/{activity_id}/streams.csv"
                r = self.session.get(url, params=q, timeout=self.timeout, auth=self._auth())
                r.raise_for_status()
                # CSV is the most consistent format → read directly
                return pd.read_csv(io.BytesIO(r.content))

            # JSON fallback: structure may be dict-of-lists. We normalize conservatively.
            url = f"{self.BASE}/activity/{activity_id}/streams"
            r = self.session.get(url, params=q, timeout=self.timeout, auth=self._auth())
            r.raise_for_status()
            data = r.json()
            # Common case is a dict mapping stream_name -> list-of-values
            if isinstance(data, dict):
                # Align lengths by index if they differ (rare). Pandas will NaN-pad.
                return pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
            # If server ever returns a list of records, normalize:
            return pd.json_normalize(data)

    @staticmethod
    def _validate_event_for_calendar(ev: Dict[str, Any]) -> None:
        # category
        if ev.get("category") not in {"WORKOUT", "NOTE", "RACE", "APPOINTMENT"}:
            raise ValueError("event.category must be one of WORKOUT|NOTE|RACE|APPOINTMENT")
        # start_date_local
        sdl = ev.get("start_date_local")
        if not (isinstance(sdl, str) and ISO_LOCAL_RE.match(sdl)):
            raise ValueError("event.start_date_local must be 'YYYY-MM-DDTHH:MM:SS' (local time, no timezone)")
        # content: require one of description OR file OR workout_id
        has_desc = bool(ev.get("description"))
        has_file = bool(ev.get("filename")) and (bool(ev.get("file_contents")) or bool(ev.get("file_contents_base64")))
        has_wid  = bool(ev.get("workout_id"))
        if not (has_desc or has_file or has_wid):
            raise ValueError("Provide description OR filename+file_contents(_base64) OR workout_id")

    def upsert_events(
        self,
        events: Iterable[Dict[str, Any]],
        athlete_id: Union[int, str] = "0",
        upsert: bool = True,
    ):
        url = f"{self.BASE}/athlete/{athlete_id}/events/bulk"
        payload = list(events)
        # local validation (nice errors before HTTP 422)
        for ev in payload:
            self._validate_event_for_calendar(ev)
        params = {"upsert": "true" if upsert else "false"}
        r = self.session.post(url, params=params, json=payload, timeout=self.timeout, auth=self._auth())
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # Surface server validation details (Intervals returns JSON/text explaining the problem)
            detail = (r.text or "")[:500]
            raise requests.HTTPError(f"{e}\nServer said: {detail}") from e
        return r.json()

    def schedule_workout(
        self,
        *,
        start_date_local: str,
        athlete_id: Union[int, str] = "0",
        name: Optional[str] = None,
        description: Optional[str] = None,
        filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        workout_id: Optional[Union[int, str]] = None,
        external_id: Optional[str] = None,
        category: str = "WORKOUT",
        use_base64: bool = True,
        **extra_fields: Any,
    ):
        ev: Dict[str, Any] = {"category": category, "start_date_local": start_date_local}
        if name:
            ev["name"] = name
        if description:
            ev["description"] = description
        if workout_id is not None:
            ev["workout_id"] = workout_id
        if filename and file_bytes:
            ev["filename"] = filename
            if use_base64:
                ev["file_contents_base64"] = base64.b64encode(file_bytes).decode("ascii")
            else:
                # textual formats like .zwo/.mrc/.erg can be sent as text; FIT should be base64
                try:
                    ev["file_contents"] = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    ev["file_contents_base64"] = base64.b64encode(file_bytes).decode("ascii")
        if external_id:
            ev["external_id"] = external_id
        ev.update(extra_fields)
        return self.upsert_events([ev], athlete_id=athlete_id, upsert=True)

    def get_events(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Optional[Union[str, date]] = None,
        newest: Optional[Union[str, date]] = None,
        category: Optional[str] = None,   # e.g., "WORKOUT"
        limit: Optional[int] = None,
        resolve: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch calendar events (planned workouts live here with category='WORKOUT').
        Returns a DataFrame; also includes convenience columns:
            - start_day (local date, floored to day)
            - duration_min_planned, distance_km_planned, load_planned
            - sport (falls back to 'type' if missing)
        """
        params: Dict[str, Any] = {}
        if oldest:
            params["oldest"] = self._fmt_date(oldest)
        if newest:
            params["newest"] = self._fmt_date(newest)
        if category:
            params["category"] = category
        if limit:
            params["limit"] = int(limit)
        if resolve:
            params["resolve"] = "true"

        url = f"{self.BASE}/athlete/{athlete_id}/events"
        r = self.session.get(url, params=params, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()

        rows = r.json()
        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows)

        # Parse times
        for col in ("start_date_local", "start_date", "created", "updated"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Local day (prefer start_date_local if present)
        ts_col = "start_date_local" if "start_date_local" in df.columns else "start_date"
        if ts_col in df.columns:
            df["start_day"] = df[ts_col].dt.floor("D")
        else:
            df["start_day"] = pd.NaT

        # Normalize sport/modality
        if "sport" not in df.columns:
            df["sport"] = df.get("type", "Other").fillna("Other")

        # Planned metrics — be liberal about possible field names
        # Duration (seconds -> minutes)
        dur_candidates = [c for c in ["planned_secs", "duration", "planned_duration"] if c in df.columns]
        if dur_candidates:
            df["duration_min_planned"] = pd.to_numeric(df[dur_candidates[0]], errors="coerce") / 60.0
        else:
            df["duration_min_planned"] = np.nan

        # Distance (m -> km)
        dist_candidates = [c for c in ["planned_distance", "distance"] if c in df.columns]
        if dist_candidates:
            df["distance_km_planned"] = pd.to_numeric(df[dist_candidates[0]], errors="coerce") / 1000.0
        else:
            df["distance_km_planned"] = np.nan

        # Load
        load_candidates = [c for c in ["planned_load", "load", "icu_training_load"] if c in df.columns]
        if load_candidates:
            df["load_planned"] = pd.to_numeric(df[load_candidates[0]], errors="coerce")
        else:
            df["load_planned"] = np.nan

        return df

    def get_athlete_profile(self, athlete_id: Union[int, str] = "0") -> pd.DataFrame:
        """Return profile for a specific athlete; '0' = self."""
        url = f"{self.BASE}/athlete/{athlete_id}"
        r = self.session.get(url, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()
        df = pd.json_normalize(r.json())
        # Ensure id is string like 'i392207'
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        # Try to form a display name if not present
        if "display_name" not in df.columns:
            parts = [df.get("firstname", pd.Series([""])).iloc[0], df.get("lastname", pd.Series([""])).iloc[0]]
            df["display_name"] = (df.get("name") or pd.Series([" ".join(p for p in parts if p)])).astype(str)
        return df

    def list_athletes(self) -> pd.DataFrame:
        """List athletes you follow/coach (API key must have access)."""
        url = f"{self.BASE}/athletes"
        r = self.session.get(url, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()
        df = pd.json_normalize(r.json())
        if df.empty:
            return df
        # Normalise common fields
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        # Heuristic for display name
        for cand in ["display_name", "name", "full_name", "athleteName", "username"]:
            if cand in df.columns:
                df["display_name"] = df[cand].astype(str)
                break
        df["display_name"] = df.get("display_name", pd.Series(["Unnamed"] * len(df))).fillna("Unnamed")
        return df


    def delete_events_range(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Optional[Union[str, date]] = None,
        newest: Optional[Union[str, date]] = None,
        category: Optional[Union[str, Iterable[str]]] = None,
        created_by_id: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Delete a range of calendar events.

        Intervals endpoint:
          DELETE /api/v1/athlete/{id}/events?oldest=...&newest=...&category=...
        Supports createdById to target only events created by a specific user/app.

        Notes:
          - 'oldest' is required by the API.
          - If 'newest' is omitted, Intervals may delete everything from 'oldest' onward.
        """
        if not oldest:
            raise ValueError("oldest is required to delete a range of events.")

        params: Dict[str, Any] = {"oldest": self._fmt_date(oldest)}
        if newest:
            params["newest"] = self._fmt_date(newest)

        if category:
            if isinstance(category, str):
                params["category"] = category
            else:
                params["category"] = ",".join(category)

        # API param appears as createdById (camelCase)
        if created_by_id is not None:
            params["createdById"] = str(created_by_id)

        url = f"{self.BASE}/athlete/{athlete_id}/events"
        r = self.session.delete(url, params=params, timeout=self.timeout, auth=self._auth())

        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            detail = (r.text or "")[:500]
            raise requests.HTTPError(f"{e}\nServer said: {detail}") from e

        # Some endpoints return JSON, some return empty/text. Be defensive.
        try:
            return r.json()
        except ValueError:
            return {"status_code": r.status_code, "text": r.text}

    def delete_planned_workouts(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Union[str, date] = None,
        newest: Optional[Union[str, date]] = None,
        *,
        dry_run: bool = True,
        created_by_id: Optional[Union[int, str]] = None,
        limit_preview: Optional[int] = 50,
    ):
        """
        Delete planned workouts (calendar events with category=WORKOUT) in a date range.

        If dry_run=True:
          - returns a DataFrame of workouts that WOULD be deleted (preview/safety).
        If dry_run=False:
          - performs the delete and returns the API response.
        """
        if not oldest:
            raise ValueError("oldest is required.")

        # Safety preview: show what you're about to delete
        if dry_run:
            df = self.get_events(
                athlete_id=athlete_id,
                oldest=oldest,
                newest=newest,
                category="WORKOUT",
                limit=limit_preview,
                resolve=False,
            )
            # If you want to narrow to only events created by your integration,
            # you can additionally filter the preview if created_by_id exists in payload fields.
            return df

        # Execute delete
        return self.delete_events_range(
            athlete_id=athlete_id,
            oldest=oldest,
            newest=newest,
            category="WORKOUT",
            created_by_id=created_by_id,
        )

    #WELLNESS
    def get_wellness(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Optional[Union[str, date]] = None,
        newest: Optional[Union[str, date]] = None,
        limit: Optional[int] = None,
        cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        List wellness records for an athlete.

        Intervals endpoints:
          GET /api/v1/athlete/{id}/wellness?oldest=...&newest=...&limit=...&cols=a,b,c

        Notes:
          - Wellness records are daily. The record id is typically an ISO date string (YYYY-MM-DD).
          - 'cols' can be used to request a subset of fields (including custom wellness fields).
        """
        params: Dict[str, Any] = {}
        if oldest:
            params["oldest"] = self._fmt_date(oldest)
        if newest:
            params["newest"] = self._fmt_date(newest)
        if limit:
            params["limit"] = int(limit)
        if cols:
            # API expects comma-separated list
            params["cols"] = ",".join([c.strip() for c in cols if c and str(c).strip()])

        url = f"{self.BASE}/athlete/{athlete_id}/wellness"
        r = self.session.get(url, params=params, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()

        rows = r.json()
        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows)

        # Common fields seen in wellness payloads
        if "id" in df.columns:
            # 'id' is usually the local date of the record
            df["day"] = pd.to_datetime(df["id"], errors="coerce").dt.date

        # Some records include timestamps like "updated"
        for col in ("updated", "created"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def get_wellness_csv(
        self,
        athlete_id: Union[int, str] = "0",
        oldest: Optional[Union[str, date]] = None,
        newest: Optional[Union[str, date]] = None,
        cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        CSV download for wellness.

        Endpoint:
          GET /api/v1/athlete/{id}/wellness.csv?oldest=...&newest=...&cols=...
        """
        params: Dict[str, Any] = {}
        if oldest:
            params["oldest"] = self._fmt_date(oldest)
        if newest:
            params["newest"] = self._fmt_date(newest)
        if cols:
            params["cols"] = ",".join([c.strip() for c in cols if c and str(c).strip()])

        url = f"{self.BASE}/athlete/{athlete_id}/wellness.csv"
        r = self.session.get(url, params=params, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()

        df = pd.read_csv(io.BytesIO(r.content))
        if "id" in df.columns:
            df["day"] = pd.to_datetime(df["id"], errors="coerce").dt.date
        return df

    def get_wellness_record(
        self,
        day: Union[str, date],
        athlete_id: Union[int, str] = "0",
    ) -> Dict[str, Any]:
        """
        Get a single wellness record by day.

        Endpoint (commonly supported):
          GET /api/v1/athlete/{id}/wellness/{YYYY-MM-DD}
        """
        day_str = self._fmt_date(day)
        url = f"{self.BASE}/athlete/{athlete_id}/wellness/{day_str}"
        r = self.session.get(url, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()
        return r.json()

    def upsert_wellness_bulk(
        self,
        records: List[Dict[str, Any]],
        athlete_id: Union[int, str] = "0",
    ) -> Any:
        """
        Bulk update wellness records.

        Endpoint:
          PUT /api/v1/athlete/{id}/wellness-bulk

        Payload:
          [
            {"id": "2024-11-20", "weight": 69.1, "restingHR": 45},
            {"id": "2024-11-19", "weight": 69.3}
          ]
        """
        # Light validation: each record must have an id (YYYY-MM-DD)
        for rec in records:
            if "id" not in rec:
                raise ValueError("Each wellness record must include an 'id' field (YYYY-MM-DD).")

        url = f"{self.BASE}/athlete/{athlete_id}/wellness-bulk"
        r = self.session.put(url, json=records, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return {"status_code": r.status_code, "text": r.text}

    def upsert_wellness_record(
        self,
        day: Union[str, date],
        payload: Dict[str, Any],
        athlete_id: Union[int, str] = "0",
    ) -> Any:
        """
        Update a single wellness record for a specific day.

        Endpoint (commonly supported):
          PUT /api/v1/athlete/{id}/wellness/{YYYY-MM-DD}

        'payload' should include only the fields you want to set/update.
        """
        day_str = self._fmt_date(day)
        url = f"{self.BASE}/athlete/{athlete_id}/wellness/{day_str}"
        r = self.session.put(url, json=payload, timeout=self.timeout, auth=self._auth())
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return {"status_code": r.status_code, "text": r.text}

# -----------------------------
# Example usage
# -----------------------------
'''
# OPTION A: Personal script with API key (Basic auth)
client = IntervalsICUClient(api_key=API_KEY)


# Pull last 90 days for the authenticated athlete
df = client.get_activities(
     athlete_id="0",               # "0" = the athlete tied to this credential
     oldest="2025-07-16",
     newest="2025-10-14",
     limit=None,                   # or e.g., 500
     format="json"                 # or "csv"
 )



activity_id = df['id'].iloc[1] 


act_df = client.get_activity(activity_id)

'''

