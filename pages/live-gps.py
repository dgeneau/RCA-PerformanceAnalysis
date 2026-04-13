# pages/live_gps.py
import os
import time
import threading
import websocket
import json
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import requests
import plotly.graph_objects as go

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
from settings import INSIDERS_USERNAME, INSIDERS_PASSWORD 
from dash import ctx

dash.register_page(__name__, path="/live-gps", name="Live GPS")

# ------------------------------------------
# Helper function for pace formatting
# ------------------------------------------
def convert_time(speed: float) -> str:
    if speed and speed > 0.5:
        seconds = 500 / speed
        minutes = int(seconds // 60)
        seconds_remainder = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:02d}"
    return "00:00.00"


# ------------------------------------------
# Data processing helpers
# ------------------------------------------
def compute_stroke_rate(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Sensor Not Found"
    if "speed" not in df.columns:
        return "0"

    accel = np.diff(df["speed"].astype(float).to_numpy()) / (1 / 10)
    stroke_peaks, _ = find_peaks(accel * -1, height=3, distance=10)

    if len(stroke_peaks) > 1:
        stroke_rate = 60 / (np.diff(stroke_peaks) / 10)
        return str(round(float(stroke_rate[-1]), 2))
    return "0"


def compute_avg_speed(df: pd.DataFrame, ref_speed: float) -> str:
    if df is None or df.empty or "speed" not in df.columns:
        return "0"

    spd = df["speed"].astype(float).to_numpy()
    accel = np.diff(spd) / (1 / 10)
    stroke_peaks, _ = find_peaks(accel * -1, height=3, distance=10)

    if len(stroke_peaks) > 1:
        start = stroke_peaks[-2]
        end = stroke_peaks[-1]
        stroke_speed = float(np.mean(spd[start:end])) if end > start else float(np.mean(spd))
        pct = round(stroke_speed / float(ref_speed) * 100, 2) if ref_speed else 0
        return f"{pct}, {round(stroke_speed, 2)}"
    return "0"


def compute_split_time(df: pd.DataFrame) -> str:
    if df is None or df.empty or "speed" not in df.columns:
        return "00:00.00"

    spd = df["speed"].astype(float).to_numpy()
    accel = np.diff(spd) / (1 / 10)
    stroke_peaks, _ = find_peaks(accel * -1, height=3, distance=10)

    if len(stroke_peaks) >= 2:
        start = stroke_peaks[-2]
        end = stroke_peaks[-1]
        mean_vel = float(np.mean(spd[start:end])) if end > start else float(np.mean(spd))
        return convert_time(mean_vel)

    return "00:00.00"


def tail_df(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) > n:
        return df.iloc[-n:].reset_index(drop=True)
    return df


def _coerce_battery_percent(value):
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            if not cleaned:
                return None
            value = float(cleaned)
        else:
            value = float(value)
    except Exception:
        return None

    # Support either ratio (0-1) or direct percent (0-100)
    if 0.0 <= value <= 1.0:
        value = value * 100.0

    if 0.0 <= value <= 100.0:
        return round(value, 1)
    return None


def _extract_battery_percent(payload):
    key_hits = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_l = str(k).lower()
                if any(token in k_l for token in ("battery", "batt", "charge", "soc")):
                    key_hits.append(v)
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)

    # Prefer the latest candidate from payload traversal
    for candidate in reversed(key_hits):
        pct = _coerce_battery_percent(candidate)
        if pct is not None:
            return pct
    return None


# ------------------------------------------
# Config (move secrets to env vars ideally)
# ------------------------------------------
client_id = "M9d3J3axpMU9z9xMfaqNtOB4BdxmsyPMK2v63yBC"
client_secret = "POh9pZ1djjNOtS8rX9FzTYHAv3ARYIvaht9pfXlKPc3axcTaCPoxYehS3OVOGhRUn9ahaujugmftpajWC7zAW0LoxVEBMhhEIg86D4Yp5g05KfT9SLGSrin6oyd0SNnd"

username = os.getenv("ASI_USERNAME", INSIDERS_USERNAME)
password = os.getenv("ASI_PASSWORD", INSIDERS_PASSWORD)

dev_dict = {
    "6499": "Baby Blue",
    "19429": "Black Betty",
    "22953": "22953", 
    "22999": "22999", 
    "23004": "23004", 
    "22993": "22993", 
    "22621": "22621"
}

EXCLUDE_DEVICE_IDS = {19429}  # optional


# ------------------------------------------
# Global store + thread control (IMPORTANT in multipage apps)
# ------------------------------------------
device_data = {}
device_battery = {}
device_lock = threading.Lock()
last_update_time = time.time()

_ws_started = False
_ws_started_lock = threading.Lock()


def _login_and_get_token() -> str:
    data_login = {"grant_type": "password", "username": username, "password": password}
    resp = requests.post(
        "https://api.asi.swiss/oauth2/token/",
        data=data_login,
        verify=False,
        allow_redirects=False,
        auth=(client_id, client_secret),
    ).json()
    return resp["access_token"]


def _fetch_device_ids(access_token: str):
    devices_list = requests.get(
        "https://api.asi.swiss/api/v1/devices/",
        params={"current": True},
        headers={"Authorization": f"Bearer {access_token}"},
        verify=False,
    ).json()

    devices = devices_list.get("results", [])
    ids = [d.get("id") for d in devices if d.get("id") is not None]
    # apply exclusion
    ids = [int(d) for d in ids if int(d) not in EXCLUDE_DEVICE_IDS]
    return ids


def on_message_factory(dev_id: int, access_token: str):
    def on_message(ws, message: str):
        global last_update_time

        try:
            parsed_message = json.loads(message)
        except Exception:
            return

        if parsed_message.get("message") == "Authorization needed":
            ws.send(json.dumps({"action": "authorize", "token": access_token}))
            return

        if parsed_message.get("message") == "Authorized":
            print(f"Authorized for device {dev_id}")
            return

        if "gnss" in parsed_message:
            gnss_data = parsed_message["gnss"]
            if not gnss_data:
                return

            gnss_df = pd.DataFrame(gnss_data)
            battery_pct = _extract_battery_percent(parsed_message)

            with device_lock:
                if dev_id not in device_data:
                    device_data[dev_id] = pd.DataFrame()
                device_data[dev_id] = pd.concat([device_data[dev_id], gnss_df], ignore_index=True)

                if battery_pct is not None:
                    device_battery[dev_id] = battery_pct

                # keep bounded
                if len(device_data[dev_id]) > 5000:
                    device_data[dev_id] = device_data[dev_id].iloc[-5000:].reset_index(drop=True)

                last_update_time = time.time()

    return on_message


def start_websocket(dev_id: int, websocket_url: str, access_token: str):
    websocket.enableTrace(False)
    headers = {"Authorization": f"Bearer {access_token}"}

    ws = websocket.WebSocketApp(
        websocket_url,
        on_open=lambda w: print(f"### WebSocket connected for device {dev_id}"),
        on_message=on_message_factory(dev_id, access_token),
        on_error=lambda w, err: print(f"[Device {dev_id}] Error: {err}"),
        on_close=lambda w, *_: print(f"### WebSocket closed for device {dev_id}"),
        header=headers,
    )
    ws.run_forever()


def start_ws_if_needed():
    """
    Ensures websocket threads start only once, even if Dash reloads
    pages or multiple users hit the route.
    """
    global _ws_started

    if _ws_started:
        return

    with _ws_started_lock:
        if _ws_started:
            return

        access_token = _login_and_get_token()
        ids = _fetch_device_ids(access_token)

        # init store
        with device_lock:
            for dev_id in ids:
                device_data.setdefault(dev_id, pd.DataFrame())
                device_battery.setdefault(dev_id, None)

        # start threads
        for dev_id in ids:
            url = f"wss://api.asi.swiss/ws/v1/preprocessed-data/{int(dev_id)}/"
            t = threading.Thread(target=start_websocket, args=(int(dev_id), url, access_token), daemon=True)
            t.start()

        _ws_started = True


# ------------------------------------------
# UI bits
# ------------------------------------------
boat_class_options = [
    {"label": "M8+", "value": "6.269592476"},
    {"label": "M4-", "value": "5.899705015"},
    {"label": "M2-", "value": "5.376344086"},
    {"label": "M4x", "value": "6.024096386"},
    {"label": "M2x", "value": "5.555555556"},
    {"label": "M1x", "value": "5.115089514"},
    {"label": "W8+", "value": "5.66572238"},
    {"label": "W4-", "value": "5.347593583"},
    {"label": "W2-", "value": "4.901960784"},
    {"label": "W4x", "value": "5.464480874"},
    {"label": "W2x", "value": "5.037783375"},
    {"label": "W1x", "value": "4.672897196"},
]

DEFAULT_PROG_REF = "6.269592476"


def make_value_box(title: str, value: str):
    return html.Div(
        style={
            "padding": "12px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
            "minWidth": "220px",
            "flex": "1",
        },
        children=[
            html.Div(title, style={"fontSize": "12px", "opacity": 0.75}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": 700}),
        ],
    )


def make_velocity_figure(df: pd.DataFrame):
    fig = go.Figure()

    if df is None or df.empty or "speed" not in df.columns:
        fig.update_layout(
            margin={"l": 20, "r": 10, "t": 20, "b": 25},
            height=170,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No speed data",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 12},
                }
            ],
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    speed = df["speed"].astype(float).to_numpy()
    window = min(120, len(speed))
    y = speed[-window:]
    x = np.arange(window)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"width": 2, "color": "#0f766e"},
            hovertemplate="Speed: %{y:.2f} m/s<extra></extra>",
        )
    )

    fig.update_layout(
        margin={"l": 40, "r": 10, "t": 12, "b": 30},
        height=170,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={
            "title": "Recent samples",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        yaxis={
            "title": "m/s",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
    )
    return fig


def make_device_card(dev_id: str, dev_name: str, rate: str, speed: str, split: str, battery: str, ref_speed_value: str, velocity_fig):
    return html.Div(
        style={
            "border": "1px solid #d1d5db",
            "borderRadius": "14px",
            "padding": "14px",
            "marginBottom": "12px",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "flexWrap": "wrap"},
                children=[
                    html.H4(f"Device: {dev_name}", style={"margin": "0"}),
                    html.Div(
                        style={"minWidth": "240px"},
                        children=[
                            html.Label("Prog reference", style={"fontSize": "12px", "opacity": 0.8}),
                            dcc.Dropdown(
                                id={"type": "prog-ref", "dev_id": dev_id},
                                options=boat_class_options,
                                value=ref_speed_value,   # default per card
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(style={"height": "10px"}),
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                children=[
                    make_value_box("Stroke Rate", rate),
                    make_value_box("Boat Speed (% Prog, m/s)", speed),
                    make_value_box("500m Split Time", split),
                    make_value_box("Battery", battery),
                ],
            ),
            html.Div(style={"height": "10px"}),
            html.Div(
                children=[
                    html.Div("Live velocity", style={"fontSize": "12px", "opacity": 0.75, "marginBottom": "4px"}),
                    dcc.Graph(
                        figure=velocity_fig,
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "170px"},
                    ),
                ]
            ),
        ],
    )


# ------------------------------------------
# Page layout (NOTE: no Dash() here)
# ------------------------------------------
layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H1("RCA Live GPS Monitoring"),

        # this triggers websocket start when page is loaded
        dcc.Store(id="live_gps_page_loaded", data=1),

        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"minWidth": "380px", "flex": "1"},
                    children=[
                        html.Label("Filter Device(s) (optional):"),
                        dcc.Dropdown(
                            id="livegps_device_id",
                            options=[],  # populated on interval
                            value=[],
                            multi=True,
                            placeholder="Leave empty to show all devices",
                        ),
                    ],
                ),
            ],
        ),

        html.Hr(),

        dcc.Interval(id="livegps_interval", interval=500, n_intervals=0),
        html.Div(id="livegps_device_cards"),
        html.Div(id="livegps_last_update", style={"marginTop": "12px", "opacity": 0.7}),
    ],
)


# ------------------------------------------
# Callbacks (Dash Pages auto-discovers them)
# ------------------------------------------
@dash.callback(
    Output("livegps_device_id", "options"),
    Output("livegps_device_id", "value"),
    Input("livegps_interval", "n_intervals"),
    Input("live_gps_page_loaded", "data"),
    State("livegps_device_id", "value"),
)
def refresh_device_dropdown(_n, _loaded, current_value):
    start_ws_if_needed()

    with device_lock:
        ids = sorted(device_data.keys())

    options = [{"label": dev_dict.get(str(d), str(d)), "value": str(d)} for d in ids]
    valid_values = {opt["value"] for opt in options}

    # Normalize current selection
    current_value = current_value or []
    # Keep only selections that still exist
    kept = [v for v in current_value if v in valid_values]

    # If user already has a valid selection, DO NOT overwrite it
    if kept:
        return options, kept

    # Keep empty selection valid so filtering is optional
    return options, []



@dash.callback(
    Output("livegps_device_cards", "children"),
    Output("livegps_last_update", "children"),
    Input("livegps_interval", "n_intervals"),
    Input("livegps_device_id", "value"),
    Input({"type": "prog-ref", "dev_id": ALL}, "value"),
)
def update_cards(_n, selected_devices, per_card_ref_values):
    global last_update_time
    start_ws_if_needed()

    selected_devices = selected_devices or []

    # Build a mapping: dev_id -> ref_speed_value (string)
    per_dev_ref = {}

    # ctx.inputs_list gives you the list of matched components for the ALL input
    # Structure depends on Dash version, but this pattern works in practice:
    try:
        matched = ctx.inputs_list[2]  # 3rd Input is the ALL prog-ref values
        # matched is a list of dicts like:
        # {"id": {"type":"prog-ref","dev_id":"6499"}, "property":"value", "value":"6.26"}
        for item, val in zip(matched, per_card_ref_values or []):
            dev_id = item["id"]["dev_id"]
            per_dev_ref[str(dev_id)] = val
    except Exception:
        per_dev_ref = {}

    with device_lock:
        if selected_devices:
            render_devices = selected_devices
        else:
            render_devices = [str(d) for d in sorted(device_data.keys())]

    cards = []
    with device_lock:
        for dev_id_str in render_devices:
            try:
                dev_id = int(dev_id_str)
            except Exception:
                continue

            # choose per-card ref if present, else default
            ref_val_str = per_dev_ref.get(dev_id_str, DEFAULT_PROG_REF)
            try:
                ref_speed = float(ref_val_str) if ref_val_str else float(DEFAULT_PROG_REF)
            except Exception:
                ref_speed = float(DEFAULT_PROG_REF)

            df = tail_df(device_data.get(dev_id, pd.DataFrame()), n=1000)

            rate = compute_stroke_rate(df)
            speed = compute_avg_speed(df, ref_speed)
            split = compute_split_time(df)
            velocity_fig = make_velocity_figure(df)
            battery_pct = device_battery.get(dev_id)
            battery = f"{battery_pct:.0f}%" if battery_pct is not None else "N/A"

            dev_name = dev_dict.get(dev_id_str, dev_id_str)

            # IMPORTANT: pass the dropdown default value used in this card
            cards.append(
                make_device_card(
                    dev_id=dev_id_str,
                    dev_name=dev_name,
                    rate=rate,
                    speed=speed,
                    split=split,
                    battery=battery,
                    ref_speed_value=ref_val_str if ref_val_str else DEFAULT_PROG_REF,
                    velocity_fig=velocity_fig,
                )
            )

        lu = last_update_time

    last_update_txt = "Last updated: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(lu))
    return cards, last_update_txt
