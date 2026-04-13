
#gps.py

from __future__ import annotations

import os

# Must be set BEFORE importing pyplot
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)



import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from weakref import ref

from more_itertools import factor
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

import numpy as np
from scipy.signal import find_peaks
import scipy as sp
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from reportlab.platypus import Image as RLImage
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

import matplotlib.pyplot as plt



from settings import (
    INSIDERS_USERNAME,
    INSIDERS_PASSWORD,
    INSIDERS_CLIENT_ID,
    INSIDERS_CLIENT_SECRET,
    INSIDERS_VERIFY_SSL,
)

# Register this file as a page
dash.register_page(
    __name__,
    path="/gps",
    name="GPS (Insiders)",
    title="GPS Sessions",
)


# -----------------------------
# Helpers
# -----------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"


def to_iso(dt: datetime) -> str:
    return dt.strftime(ISO_FMT)


def chunk_time_range(start: datetime, stop: datetime, chunk_hours: int = 24) -> List[Tuple[datetime, datetime]]:
    """Chunk [start, stop] into smaller intervals to avoid interval-size limits."""
    out: List[Tuple[datetime, datetime]] = []
    cur = start
    step = timedelta(hours=chunk_hours)
    while cur < stop:
        nxt = min(cur + step, stop)
        out.append((cur, nxt))
        cur = nxt
    return out


def parse_iso(s: str) -> datetime:
    return pd.to_datetime(s, utc=True).to_pydatetime()


def lowpass(signal, highcut, frequency):
    '''
    Apply a low-pass filter using Butterworth design

    Inputs;
    signal = array-like input
    high-cut = desired cutoff frequency
    frequency = data sample rate in Hz

    Returns;
    filtered signal
    '''
    order = 2
    nyq = 0.5 * frequency
    highcut = highcut / nyq #normalize cutoff frequency
    b, a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
    y = sp.signal.filtfilt(b, a, signal, axis=0)
    return y

def speed_to_split(speed):
    seconds = 500/speed
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"

def sec_to_split(seconds):
    
    # Calculate the minutes, seconds, and milliseconds
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    # Format the string with leading zeros if necessary
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}"


def elite_gnss_repair(
    df: pd.DataFrame,
    t_sec_col: str = "_t_sec",
    v_col: str = "_v",
    fs: float = 10.0,
    small_gap_s: float = 1,#0.6,     # <= 0.6s: shape-preserving cubic (PCHIP)
    medium_gap_s: float = 3,#2.0,    # <= 2.0s: linear (safer than cubic for longer gaps)
    max_fill_s: float = 5.0,      # <= 5.0s: allow fill but mark low confidence
    max_accel: float = 8.0,       # m/s^2 clamp (tune per sport; 6–10 typical)
    smooth_window_s: float = 0.5, # Savitzky-Golay window ~0.5s
    smooth_poly: int = 3,         # 2–3 is usually enough
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      repaired_df: regular 10 Hz timeline with raw/fill/smoothed + flags
      report: dropout + quality metrics
    """

    if df.empty:
        return df.copy(), {"error": "empty dataframe"}

    # --- clean/sort ---
    d = df[[t_sec_col, v_col]].copy()
    d = d.dropna(subset=[t_sec_col, v_col])
    d[t_sec_col] = pd.to_numeric(d[t_sec_col], errors="coerce")
    d[v_col] = pd.to_numeric(d[v_col], errors="coerce")
    d = d.dropna(subset=[t_sec_col, v_col]).sort_values(t_sec_col)

    # Remove duplicate timestamps (keep first)
    d = d[~d[t_sec_col].duplicated(keep="first")].reset_index(drop=True)

    if len(d) < 5:
        return d.copy(), {"error": "not enough points to repair", "n": int(len(d))}

    dt = 1.0 / fs
    t0 = float(d[t_sec_col].iloc[0])
    t1 = float(d[t_sec_col].iloc[-1])

    # Build perfect 10 Hz grid (include endpoint tolerance)
    t_grid = np.arange(t0, t1 + 0.5 * dt, dt)
    grid = pd.DataFrame({t_sec_col: t_grid})

    # Merge raw onto grid using nearest-match within half a sample
    # (because devices can jitter around exact 0.1s)
    # Approach: round times to nearest tick index
    d_tick = np.round((d[t_sec_col] - t0) / dt).astype(int)
    g_tick = np.arange(len(t_grid))

    d2 = pd.DataFrame({
        "tick": d_tick,
        t_sec_col: d[t_sec_col].values,
        "v_raw": d[v_col].values
    }).drop_duplicates("tick", keep="first")

    grid["tick"] = g_tick
    merged = grid.merge(d2[["tick", "v_raw"]], on="tick", how="left")
    merged.drop(columns=["tick"], inplace=True)

    merged["is_observed"] = ~merged["v_raw"].isna()

    # --- dropout metrics on observed timestamps ---
    obs_t = d[t_sec_col].values
    obs_dt = np.diff(obs_t)
    gap_mask = obs_dt > (1.5 * dt)
    gap_lengths = obs_dt[gap_mask] if gap_mask.any() else np.array([])

    missing_n_est = np.maximum(0, np.round(obs_dt / dt).astype(int) - 1)
    missing_total_est = int(missing_n_est.sum())

    # --- fill gaps with tiered methods ---
    merged["v_fill"] = merged["v_raw"].copy()
    merged["fill_method"] = np.where(merged["is_observed"], "observed", "missing")
    merged["confidence"] = np.where(merged["is_observed"], 1.0, np.nan)

    # Indices of observed points on the grid
    idx_obs = np.where(merged["is_observed"].values)[0]
    t_obs = merged[t_sec_col].values[idx_obs]
    v_obs = merged["v_raw"].values[idx_obs]

    # 1) Small gaps: PCHIP (shape-preserving cubic, good for speed)
    # 2) Medium gaps: linear
    # 3) Longer (<= max_fill_s): linear but low confidence
    # 4) > max_fill_s: leave NaN

    # First do a global linear fill for convenience (we'll override small gaps with PCHIP)
    merged["v_lin"] = merged["v_raw"].interpolate(method="linear", limit_direction="both")

    # Identify contiguous missing runs on the grid
    is_miss = merged["v_raw"].isna().values
    runs = []
    i = 0
    n = len(is_miss)
    while i < n:
        if is_miss[i]:
            j = i
            while j < n and is_miss[j]:
                j += 1
            runs.append((i, j - 1))  # inclusive
            i = j
        else:
            i += 1

    # Helper: fill run between two observed neighbors
    def fill_run(i0: int, i1: int):
        # Neighbors
        left = i0 - 1
        right = i1 + 1
        if left < 0 or right >= len(merged):
            return  # can't bracket

        if not merged["is_observed"].iloc[left] or not merged["is_observed"].iloc[right]:
            return

        tL = float(merged[t_sec_col].iloc[left])
        tR = float(merged[t_sec_col].iloc[right])
        gap_s = tR - tL

        # If the gap is too big, don't fill
        if gap_s > max_fill_s:
            return

        # Fill target indices in run
        idx = np.arange(i0, i1 + 1)
        t_run = merged[t_sec_col].values[idx]

        if gap_s <= small_gap_s and len(t_obs) >= 4:
            # PCHIP over all observed points for smooth-but-shape-preserving fill
            p = PchipInterpolator(t_obs, v_obs, extrapolate=False)
            v_new = p(t_run)

            merged.loc[idx, "v_fill"] = v_new
            merged.loc[idx, "fill_method"] = "pchip"
            merged.loc[idx, "confidence"] = 0.9

        elif gap_s <= medium_gap_s:
            # Linear for medium gaps
            merged.loc[idx, "v_fill"] = merged.loc[idx, "v_lin"].values
            merged.loc[idx, "fill_method"] = "linear"
            merged.loc[idx, "confidence"] = 0.7

        else:
            # Longer but allowed: linear, low confidence
            merged.loc[idx, "v_fill"] = merged.loc[idx, "v_lin"].values
            merged.loc[idx, "fill_method"] = "linear_lowconf"
            merged.loc[idx, "confidence"] = 0.4

    for (a, b) in runs:
        fill_run(a, b)

    # Any remaining NaNs after allowed fill stay NaN
    merged.drop(columns=["v_lin"], inplace=True)

    # --- acceleration clamp (optional but powerful) ---
    # Compute accel on filled signal where available
    v = merged["v_fill"].values.astype(float)
    t = merged[t_sec_col].values.astype(float)

    valid = ~np.isnan(v)
    if valid.sum() > 10:
        # gradient accel
        a = np.full_like(v, np.nan)
        a[valid] = np.gradient(v[valid], t[valid])

        # clamp unrealistic accel by lightly nudging v toward feasibility
        bad = np.where(np.abs(a) > max_accel)[0]
        if len(bad) > 0:
            # Simple strategy: apply a mild smoothing pass only around bad points
            # (keeps most of the trace intact)
            v2 = v.copy()
            for k in bad:
                lo = max(0, k - 3)
                hi = min(len(v2) - 1, k + 3)
                window = v2[lo:hi + 1]
                if np.isfinite(window).sum() >= 3:
                    v2[k] = np.nanmedian(window)
            merged["v_fill"] = v2

    # --- sprint-friendly smoothing (Savitzky-Golay) ---
    # Apply only where we have contiguous valid data
    v = merged["v_fill"].values.astype(float)
    valid = np.isfinite(v)

    # choose odd window length in samples
    win = int(round(smooth_window_s * fs))
    if win < 5:
        win = 5
    if win % 2 == 0:
        win += 1

    v_smooth = np.full_like(v, np.nan)

    # smooth contiguous valid blocks only (avoid bridging NaN gaps)
    start = 0
    while start < len(v):
        if not valid[start]:
            start += 1
            continue
        end = start
        while end < len(v) and valid[end]:
            end += 1

        block = v[start:end]
        if len(block) >= win:
            v_smooth[start:end] = savgol_filter(block, window_length=win, polyorder=smooth_poly, mode="interp")
        else:
            v_smooth[start:end] = block  # too short to smooth

        start = end

    merged["v_smooth"] = v_smooth

    # --- quality scoring ---
    n_grid = len(merged)
    n_obs = int(merged["is_observed"].sum())
    n_missing_grid = int((~merged["is_observed"]).sum())
    n_filled = int(np.isfinite(merged["v_fill"]).sum() - n_obs)
    n_unfilled = int(n_missing_grid - n_filled)

    missing_pct_grid = (n_missing_grid / n_grid) * 100.0
    filled_pct_missing = (n_filled / max(1, n_missing_grid)) * 100.0

    max_gap = float(gap_lengths.max()) if gap_lengths.size else 0.0
    n_gaps = int(gap_mask.sum())

    # Score heuristic: start 100, penalize missing %, big max gap, unfilled %
    score = 100.0
    score -= 0.8 * missing_pct_grid
    score -= 8.0 * (max_gap / 1.0)  # 8 points per second of max gap
    score -= 20.0 * (n_unfilled / max(1, n_missing_grid))
    score -= 2.0 * max(0, n_gaps - 5)  # extra gaps beyond 5

    score = float(np.clip(score, 0.0, 100.0))

    report = {
        "fs_hz": fs,
        "dt_expected": dt,
        "n_grid": int(n_grid),
        "n_observed": int(n_obs),
        "missing_pct_grid": float(missing_pct_grid),
        "n_gaps_observed_timebase": int(n_gaps),
        "max_gap_seconds": float(max_gap),
        "missing_samples_est_from_obs_dt": int(missing_total_est),
        "n_filled": int(n_filled),
        "n_unfilled": int(n_unfilled),
        "filled_pct_of_missing_grid": float(filled_pct_missing),
        "quality_score_0_100": score,
        "params": {
            "small_gap_s": small_gap_s,
            "medium_gap_s": medium_gap_s,
            "max_fill_s": max_fill_s,
            "max_accel": max_accel,
            "smooth_window_s": smooth_window_s,
            "smooth_poly": smooth_poly,
        },
    }

    return merged, report


def metric_card(title: str, value: str, subtitle: str = "", color: str = "light"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.Div(value, className="h4 mb-0"),
                html.Div(subtitle, className="text-muted small") if subtitle else None,
            ]
        ),
        className="h-100",
        color=color,
        outline=True,
    )

def parse_gnss_timestamp_to_tsec(ts: pd.Series) -> pd.Series:
    """
    Convert gnss.timestamp to seconds-from-start robustly.
    Handles:
      - ISO strings
      - epoch seconds (≈1e9)
      - epoch milliseconds (≈1e12)
      - relative seconds (small numbers)
    Returns float seconds from start.
    """
    # If it's already datetime-like strings
    if ts.dtype == "O" or pd.api.types.is_string_dtype(ts):
        t = pd.to_datetime(ts, utc=True, errors="coerce")
        t = t.sort_values()
        return (t - t.iloc[0]).dt.total_seconds()

    # Numeric timestamps
    x = pd.to_numeric(ts, errors="coerce")
    x = x.astype(float)

    med = np.nanmedian(x.values)
    if not np.isfinite(med):
        return pd.Series([np.nan] * len(ts), index=ts.index)

    # Case A: already seconds from start (or small relative seconds)
    if med < 1e6:
        # treat as seconds (relative)
        return x - np.nanmin(x.values)

    # Case B: epoch milliseconds
    if med > 1e11:
        t = pd.to_datetime(x, unit="ms", utc=True, errors="coerce")
        t = t.sort_values()
        return (t - t.iloc[0]).dt.total_seconds()

    # Case C: epoch seconds
    t = pd.to_datetime(x, unit="s", utc=True, errors="coerce")
    t = t.sort_values()
    return (t - t.iloc[0]).dt.total_seconds()

def build_stroke_data(sub: pd.DataFrame, peaks: np.ndarray, prog_key: str = "M8+") -> pd.DataFrame:
    """
    Build per-stroke metrics using peak indices.
    sub must contain: _t_sec, v_fill, _A
    peaks are indices into sub.
    """
    if peaks is None or len(peaks) < 2:
        return pd.DataFrame()

    # Ensure monotonic time and numeric
    s = sub.copy()
    s["_t_sec"] = pd.to_numeric(s["_t_sec"], errors="coerce")
    s["v_fill"] = pd.to_numeric(s["v_fill"], errors="coerce")
    s["_A"] = pd.to_numeric(s["_A"], errors="coerce")
    s = s.dropna(subset=["_t_sec"]).sort_values("_t_sec").reset_index(drop=True)

    ref = float(prog_dict.get(prog_key, prog_dict["M8+"]))

    rows = []
    cum_dist = 0.0

    for i in range(1, len(peaks)):
        a = int(peaks[i - 1])
        b = int(peaks[i])
        if b <= a + 1:
            continue

        t0 = float(s["_t_sec"].iloc[a])
        t1 = float(s["_t_sec"].iloc[b])
        dt = t1 - t0
        if not np.isfinite(dt) or dt <= 0:
            continue

        v_seg = s["v_fill"].iloc[a:b].to_numpy(dtype=float)
        a_seg = s["_A"].iloc[a:b].to_numpy(dtype=float)

        v_ok = np.isfinite(v_seg)
        a_ok = np.isfinite(a_seg)

        if v_ok.sum() < 2:
            continue

        v_mean = float(np.nanmean(v_seg))
        v_min  = float(np.nanmin(v_seg))
        v_max  = float(np.nanmax(v_seg))
        a_mean = float(np.nanmean(a_seg[a_ok])) if a_ok.any() else np.nan

        dps = v_mean * dt
        dist_start = cum_dist
        cum_dist += dps
        dist_end = cum_dist
        dist_mid = 0.5 * (dist_start + dist_end)

        stroke_rate_spm = 60.0 / dt

        rows.append({
            "stroke_idx": i,
            "t_start": t0,
            "t_end": t1,
            "dt": dt,
            "stroke_rate_spm": stroke_rate_spm,
            "speed_mean": v_mean,
            "speed_min": v_min,
            "speed_max": v_max,
            "accel_mean": a_mean,
            "DPS": dps,
            "dist_start": dist_start,
            "dist_end": dist_end,
            "dist_mid": dist_mid,
            "Prog": (v_mean / ref) * 100.0 if np.isfinite(ref) and ref > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def trapz_distance_m(t_sec: np.ndarray, speed_mps: np.ndarray) -> float:
    """Distance in meters from continuous speed trace."""
    t = np.asarray(t_sec, dtype=float)
    v = np.asarray(speed_mps, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2:
        return float("nan")
    t = t[ok]
    v = v[ok]
    # ensure monotonic
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    return float(np.trapz(v, t))

def summarize_by_distance_bins(
    sub: pd.DataFrame,
    bin_m: float = 100.0,
    stroke_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Segment summary table (25–250m in 5m steps).
    Uses CROPPED continuous speed trace for speed/splits/min/max,
    and stroke_data for Rate/DPS/Prog.
    """
    s = sub.copy()
    s["_t_sec"] = pd.to_numeric(s["_t_sec"], errors="coerce")
    s["v_fill"] = pd.to_numeric(s["v_fill"], errors="coerce")
    s = s.dropna(subset=["_t_sec", "v_fill"]).sort_values("_t_sec").reset_index(drop=True)
    if len(s) < 2:
        return pd.DataFrame()

    # continuous distance (m) from cropped trace
    t = s["_t_sec"].to_numpy(dtype=float)
    v = s["v_fill"].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    s["_dist_m"] = np.cumsum(v * dt)

    total_dist = float(s["_dist_m"].iloc[-1])
    if not np.isfinite(total_dist) or total_dist <= 0:
        return pd.DataFrame()

    # bin index: 0..N-1
    s["_bin"] = (s["_dist_m"] // bin_m).astype(int)

    out_rows = []
    for b, g in s.groupby("_bin"):
        end_m = int((b + 1) * bin_m)
        dist_label = f"{end_m}m"

        avg_speed = float(np.nanmean(g["v_fill"]))
        min_speed = float(np.nanmin(g["v_fill"]))
        max_speed = float(np.nanmax(g["v_fill"]))

        # rounding rules
        avg_speed_r = round(avg_speed, 1)
        min_speed_r = round(min_speed, 1)
        max_speed_r = round(max_speed, 1)

        split_500 = 500.0 / avg_speed if avg_speed > 0 else np.nan

        row = {
            "Distance (m)": dist_label,
            "Speed (m/s)": avg_speed_r,
            "Split (/500m)": sec_to_split(split_500) if np.isfinite(split_500) else "",
            "Min Speed (m/s)": min_speed_r,
            "Max Speed (m/s)": max_speed_r,
            "Rate (SPM)": "",
            "DPS (m)": "",
            "Prog (%)": "",
        }

        # Fill Rate/DPS/Prog from stroke_df (aka stroke_data parameter)
        if stroke_data is not None and not stroke_data.empty and "dist_mid" in stroke_data.columns:
            lo = b * bin_m
            hi = (b + 1) * bin_m
            sd = stroke_data[(stroke_data["dist_mid"] >= lo) & (stroke_data["dist_mid"] < hi)]

            if len(sd):
                rate = float(np.nanmean(sd["stroke_rate_spm"])) if "stroke_rate_spm" in sd.columns else np.nan
                dps  = float(np.nanmean(sd["DPS"])) if "DPS" in sd.columns else np.nan
                prog = float(np.nanmean(sd["Prog"])) if "Prog" in sd.columns else np.nan

                # rounding rules
                row["Rate (SPM)"] = f"{int(round(rate))}" if np.isfinite(rate) else ""
                row["DPS (m)"]    = f"{dps:.2f}" if np.isfinite(dps) else ""
                row["Prog (%)"] = f"{int(round(prog))}" if np.isfinite(prog) else ""

        out_rows.append(row)

    return pd.DataFrame(out_rows)

def build_race_report_pdf_bytes(
    sub: pd.DataFrame,
    stroke_df: pd.DataFrame | None = None,
    title: str = "Race Report",
    segment_m: float = 100.0,
) -> bytes:
    """
    Returns a PDF as bytes.
    sub must contain: _t_sec, v_fill (cropped)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=12*mm, bottomMargin=12*mm
    )
    styles = getSampleStyleSheet()
    story = []

    # --- Overall from CROPPED continuous trace ---
    t0 = float(sub["_t_sec"].iloc[0])
    t1 = float(sub["_t_sec"].iloc[-1])
    race_time_s = max(0.0, t1 - t0)

    race_dist_m = trapz_distance_m(sub["_t_sec"].to_numpy(), sub["v_fill"].to_numpy())

    avg_speed = float(np.nanmean(sub["v_fill"]))
    avg_split_s = 500.0 / avg_speed if avg_speed > 0 else np.nan

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Race Distance: {race_dist_m:.0f} m", styles["Normal"]))
    story.append(Paragraph(f"Race Time (GPS): {sec_to_split(race_time_s)}", styles["Normal"]))
    story.append(Paragraph(f"Average Speed: {avg_speed:.2f} m/s", styles["Normal"]))
    story.append(Paragraph(f"Average Split (/500m): {sec_to_split(avg_split_s) if np.isfinite(avg_split_s) else ''}", styles["Normal"]))
    story.append(Spacer(1, 8))

    # --- 100m table like your example ---
    bins = summarize_by_distance_bins(sub, bin_m=float(segment_m), stroke_data=stroke_df)


    # OPTIONAL: if you want rate/DPS/Prog columns populated from stroke_data,
    # you can aggregate stroke_data into 100m bins and merge here.

    if not bins.empty:
        header = list(bins.columns)
        table_data = [header] + bins.values.tolist()

        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,0), (-1,-1), "RIGHT"),
            ("ALIGN", (0,0), (0,-1), "LEFT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    # --- Speed & Rate vs Distance plot (instead of 2nd table) ---
    if stroke_df is not None and not stroke_df.empty:
        story.append(Paragraph("<b>Per-stroke Speed & Rate vs Distance</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))

        png = stroke_plot_png_bytes(stroke_df)
        if png:
            img = RLImage(io.BytesIO(png))
            img.drawWidth = 180 * mm
            img.drawHeight = 90 * mm
            story.append(img)
            story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()

def stroke_plot_png_bytes(stroke_df: pd.DataFrame) -> bytes:
    if stroke_df is None or stroke_df.empty:
        return b""

    # Use dist_end or dist_mid safely
    xcol = "dist_end" if "dist_end" in stroke_df.columns else "dist_mid"
    x = stroke_df[xcol].to_numpy(dtype=float)
    speed = np.round(stroke_df["speed_mean"].to_numpy(dtype=float), 1)
    rate = np.round(stroke_df["stroke_rate_spm"].to_numpy(dtype=float)).astype(int)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax2 = ax1.twinx()

    # SPEED (black line)
    ax1.plot(
        x,
        speed,
        linewidth=2,
        color="black",
        label="Speed"
    )

    # STROKE RATE (red line)
    ax2.plot(
        x,
        rate,
        linewidth=2,
        color="red",
        label="Stroke Rate"
    )

    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Speed (m/s)", color="black")
    ax2.set_ylabel("Stroke Rate (spm)", color="red")

    ax2.set_ylim(10, 50)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()


    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()





# -----------------------------
# OAuth2 token (password grant)
# -----------------------------
def get_access_token_password_grant(
    username: str,
    password: str,
    client_id: str,
    client_secret: str,
    verify_ssl: bool = True,
) -> str:
    data_login = {"grant_type": "password", "username": username, "password": password}

    r = requests.post(
        "https://api.asi.swiss/oauth2/token/",
        data=data_login,
        verify=verify_ssl,  # you can disable via toggle (not recommended)
        allow_redirects=False,
        auth=(client_id, client_secret),
        timeout=60,
    )
    r.raise_for_status()
    response = r.json()

    if "access_token" not in response:
        raise RuntimeError(f"No access_token returned. Response keys: {list(response.keys())}")

    return response["access_token"]


# -----------------------------
# Insiders API client
# -----------------------------
class InsidersClient:
    def __init__(self, access_token: str, base_url: str = "https://api.insiders.live"):
        self.access_token = access_token
        self.base_url = base_url

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.get(url, headers=self._headers(), params=params, timeout=60)

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "5"))
            time.sleep(retry_after)
            r = requests.get(url, headers=self._headers(), params=params, timeout=60)

        r.raise_for_status()
        return r.json()

    def list_devices(self, limit: int = 3000) -> List[Dict[str, Any]]:
        page = 1
        out: List[Dict[str, Any]] = []
        while True:
            data = self._get("/v1/devices/", params={"page": page, "limit": limit})
            out.extend(data.get("results", []))
            if not data.get("next"):
                break
            page = data["next"]
        return out

    def list_preprocessed_ranges(self, device_id: int, start: str, stop: str, limit: int = 3000) -> List[Dict[str, Any]]:
        page = 1
        out: List[Dict[str, Any]] = []
        while True:
            data = self._get(
                f"/v1/devices/{device_id}/preprocessed-ranges/",
                params={"start": start, "stop": stop, "page": page, "limit": limit},
            )
            out.extend(data.get("results", []))
            if not data.get("next"):
                break
            page = data["next"]
        return out

    def fetch_preprocessed_data(self, device_id: int, start: str, stop: str, limit: int = 25000) -> pd.DataFrame:
        page = 1
        rows: List[Dict[str, Any]] = []
        while True:
            data = self._get(
                f"/v1/devices/{device_id}/preprocessed/",
                params={"start": start, "stop": stop, "page": page, "limit": limit},
            )
            rows.extend(data.get("results", []))
            if not data.get("next"):
                break
            page = data["next"]
        return pd.DataFrame(rows)



def safe_device_label(d: Dict[str, Any]) -> str:
    name = d.get("name") or d.get("label") or d.get("serial_number") or f"Device {d.get('id')}"
    return f"{name} (id={d.get('id')})"


def session_label(sess: Dict[str, Any]) -> str:
    start = parse_iso(sess["start"])
    stop = parse_iso(sess["stop"])
    dur = stop - start
    mins = max(0, int(dur.total_seconds() // 60))
    return f"Device {sess['device_id']} • {start.strftime('%Y-%m-%d %H:%M UTC')} → {stop.strftime('%H:%M UTC')} ({mins} min)"




prog_dict = {
            "M8+":"6.269592476",
            "M4-":"5.899705015",
            "M2-":"5.376344086",
            "M4x":"6.024096386",
            "M2x":"5.555555556",
            "M1x":"5.115089514",
            "W8+":"5.66572238",
            "W4-":"5.347593583",
            "W2-":"4.901960784",
            "W4x":"5.464480874",
            "W2x":"5.037783375",
            "W1x":"4.672897196",
                }

# -----------------------------
# Page layout
# -----------------------------
layout = dbc.Container(
    [
        html.H3("Insiders GPS Sessions", className="mt-3"),

        # Login
        html.Div(id="gps-login-status", className="small mb-2"),
        # Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Lookback window (days)"),
                        dcc.Slider(
                            id="gps-lookback-days",
                            min=1,
                            max=14,
                            step=1,
                            value=7,
                            marks={i: str(i) for i in [1, 3, 7, 14]},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Anchor date "),
                        dcc.DatePickerSingle(
                            id="gps-anchor-date",
                            date=datetime.now().date(),
                            display_format="YYYY-MM-DD",
                            clearable=False,
                        ),
                        dbc.FormText("Search window ends on this date."),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Device"),
                        dcc.Dropdown(
                            id="gps-device-dropdown",
                            options=[],
                            placeholder="Login first to load devices...",
                            clearable=True,
                        ),
                        dbc.FormText("Optional: leave blank to load sessions across ALL devices (can be slower)."),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("\u00A0"),
                        dbc.Button("Load Sessions", id="gps-btn-load-sessions", color="secondary", className="w-100"),
                        html.Div(id="gps-load-status", className="mt-2 small"),
                    ],
                    md=2,
                ),
            ],
            className="mb-3",
        ),


        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Session"),
                        dcc.Dropdown(
                            id="gps-session-dropdown",
                            options=[],
                            placeholder="Click 'Load Sessions' first...",
                            clearable=False,
                        ),
                        dbc.FormText("Sessions are contiguous blocks (gap ≤ 5 minutes)."),
                    ],
                    md=9,
                ),


                dbc.Col(
                    [
                        dbc.Label("Summary segment (m)"),
                        dcc.Slider(
                            id="gps-seg-m",
                            min=25,
                            max=250,
                            step=5,
                            value=100,
                            marks={25: "25", 50: "50", 100: "100", 150: "150", 200: "200", 250: "250"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        dbc.FormText("Used for distance summary rows and PDF report."),
                    ],
                    md=3,
                ),


                dbc.Col(
                    [
                        dbc.Label("Prog reference"),
                        dcc.Dropdown(
                            id="gps-prog-dropdown",
                            options=[{"label": k, "value": k} for k in prog_dict.keys()],
                            value="M8+",
                            clearable=False,
                        ),
                        dbc.FormText("Used for Prog (%) metric."),
                    ],
                    md=3,
                ),


                dbc.Col(
                    [
                        dbc.Label("\u00A0"),
                        dbc.Button("Pull & Plot", id="gps-btn-pull-plot", color="success", className="w-100"),
                        html.Div(id="gps-pull-status", className="mt-2 small"),
                    ],
                    md=3,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(
                        dcc.Graph(id="gps-velo-plot", figure=go.Figure(), config={"displayModeBar": True}),
                        type="default",
                    ),
                    md=12,
                )
            ]
        ),
        dcc.Store(id="gps-session-df-store"),   # store pulled session data (smallish or downsampled)
        dcc.Store(id="gps-crop-store"),         # store crop start/end + indices

        html.Div(id="gps-crop-status", className="small mt-2"),

        dcc.Graph(id="gps-velo-plot-cropped"),
        html.Hr(className = "my-3"), 
        html.Div(id = 'gps-metrics-cards'),
                dcc.Interval(
                    id="gps-auto-auth",
                    interval=250,   # ms
                    n_intervals=0,
                    max_intervals=1 # run only once
                ),
        dcc.Store(id="gps-stroke-store"),
        dcc.Graph(id="gps-stroke-metrics-plot"), 

        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Download PDF Report", id="gps-btn-download-pdf", color="primary", className="w-100"),
                    md=3,
                ),
                dbc.Col(html.Div(id="gps-pdf-status", className="small"), md=9),
            ],
            className="mt-3",
        ),
        dcc.Download(id="gps-download-pdf"),


        # Stores (page-scoped IDs)
        dcc.Store(id="gps-token-store"),     # {"access_token": "..."}
        dcc.Store(id="gps-devices-store"),   # list of device dicts
        dcc.Store(id="gps-sessions-store"),  # list of session dicts
    ],
    fluid=True,
)


# -----------------------------
# Callbacks
# -----------------------------
@dash.callback(
    Output("gps-token-store", "data"),
    Output("gps-login-status", "children"),
    Input("gps-auto-auth", "n_intervals"),
    State("gps-token-store", "data"),
)
def gps_auto_login(n_intervals, existing_token):
    # If we already have a token in the store, don't re-auth
    if existing_token and "access_token" in existing_token:
        return no_update, "✅ Authenticated."

    try:
        token = get_access_token_password_grant(
            username=INSIDERS_USERNAME,
            password=INSIDERS_PASSWORD,
            client_id=INSIDERS_CLIENT_ID,
            client_secret=INSIDERS_CLIENT_SECRET,
            verify_ssl=INSIDERS_VERIFY_SSL,
        )
        return {"access_token": token}, "✅ Authenticated (auto)."
    except Exception as e:
        return None, f"Auto-auth error: {e}"


@dash.callback(
    Output("gps-device-dropdown", "options"),
    Output("gps-device-dropdown", "placeholder"),
    Output("gps-devices-store", "data"),
    Input("gps-token-store", "data"),
)
def gps_load_devices(token_data):
    if not token_data or "access_token" not in token_data:
        return [], "Login first to load devices...", None

    try:
        c = InsidersClient(access_token=token_data["access_token"])
        devices = c.list_devices()

        # 🚫 EXCLUDE BAD DEVICE
        exclude_ids = {19429, 6499, 22662}
        devices = [d for d in devices if int(d.get("id", -1)) not in exclude_ids]

        opts = [{"label": safe_device_label(d), "value": d["id"]} for d in devices]
        return opts, "Select a device (optional)", devices

    except Exception as e:
        return [], f"Error loading devices: {e}", None


@dash.callback(
    Output("gps-sessions-store", "data"),
    Output("gps-session-dropdown", "options"),
    Output("gps-session-dropdown", "value"),
    Output("gps-load-status", "children"),
    Input("gps-btn-load-sessions", "n_clicks"),
    State("gps-token-store", "data"),
    State("gps-devices-store", "data"),
    State("gps-lookback-days", "value"),
    State("gps-anchor-date", "date"),   # NEW
    State("gps-device-dropdown", "value"),
    prevent_initial_call=True,
)

def gps_load_sessions(n_clicks, token_data, devices_data, lookback_days, anchor_date, device_id):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    if not token_data or "access_token" not in token_data:
        return None, [], None, "Please login first."

    now = datetime.now(timezone.utc)

    # Anchor date from DatePickerSingle comes as 'YYYY-MM-DD'
    anchor = pd.to_datetime(anchor_date).date() if anchor_date else datetime.now().date()

    # Use end-of-day UTC for window_stop to include the whole day
    window_stop = datetime(anchor.year, anchor.month, anchor.day, 23, 59, 59, tzinfo=timezone.utc)
    window_start = window_stop - timedelta(days=int(lookback_days))

    if device_id:
        device_ids = [int(device_id)]
    else:
        if not devices_data:
            return None, [], None, "No devices loaded. Login again."
        device_ids = [int(d["id"]) for d in devices_data]

    c = InsidersClient(access_token=token_data["access_token"])

    try:
        all_sessions: List[Dict[str, Any]] = []
        for did in device_ids:
            ranges = c.list_preprocessed_ranges(did, to_iso(window_start), to_iso(window_stop))
            for r in ranges:
                all_sessions.append({"device_id": did, "start": r["start"], "stop": r["stop"]})

        all_sessions.sort(key=lambda s: s["start"], reverse=True)

        options = [{"label": session_label(s), "value": i} for i, s in enumerate(all_sessions)]
        status = f"Found {len(all_sessions)} sessions in the last {lookback_days} day(s)."
        value = 0 if options else None
        return all_sessions, options, value, status

    except Exception as e:
        return None, [], None, f"Error loading sessions: {e}"


@dash.callback(
    Output("gps-velo-plot", "figure"),
    Output("gps-pull-status", "children"),
    Output("gps-session-df-store", "data"),   
    Input("gps-btn-pull-plot", "n_clicks"),
    State("gps-token-store", "data"),
    State("gps-sessions-store", "data"),
    State("gps-session-dropdown", "value"),
    prevent_initial_call=True,
)
def gps_pull_and_plot(n_clicks, token_data, sessions_data, selected_idx):
    if not n_clicks:
        return no_update, no_update, no_update
    if not token_data or "access_token" not in token_data:
        return no_update, "Please login first."
    if not sessions_data or selected_idx is None:
        return no_update, "No session selected."

    sess = sessions_data[int(selected_idx)]
    device_id = int(sess["device_id"])
    sess_start = parse_iso(sess["start"])
    sess_stop = parse_iso(sess["stop"])

    c = InsidersClient(access_token=token_data["access_token"])

    fig = go.Figure()
    fig.update_layout(
        title=f"Velocity vs Time • Device {device_id}",
        xaxis_title="Time (UTC)",
        yaxis_title="Velocity",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    try:
        df_parts = []
        for a, b in chunk_time_range(sess_start, sess_stop, chunk_hours=24):
            df_parts.append(c.fetch_preprocessed_data(device_id, to_iso(a), to_iso(b)))
        df = pd.concat(df_parts, ignore_index=True) if df_parts else pd.DataFrame()

        if df.empty:
            return fig, "Session pulled, but returned 0 points.", None

        time_col = "gnss.timestamp"
        vel_col  = "gnss.speed"

                
        if time_col not in df.columns or vel_col not in df.columns:
            cols = ", ".join(df.columns[:30])
            return fig, f"Missing {time_col} or {vel_col}. Columns: {cols}", None

        
        # ---- Build GNSS-only frame (time + speed) ----
        gnss = df[[time_col, vel_col]].copy()
        gnss[vel_col] = pd.to_numeric(gnss[vel_col], errors="coerce")
        gnss = gnss.dropna(subset=[time_col, vel_col]).reset_index(drop=True)

        # ---- Robust time -> seconds from start ----
        gnss["_t_sec"] = parse_gnss_timestamp_to_tsec(gnss[time_col])
        gnss = gnss.dropna(subset=["_t_sec"]).sort_values("_t_sec").reset_index(drop=True)

        # Speed (m/s)
        gnss["_v"] = gnss[vel_col].astype(float)

        # ---- sanity check to catch the "one point" issue early ----
        n_unique = int(gnss["_t_sec"].nunique())
        dur = float(gnss["_t_sec"].iloc[-1]) if len(gnss) else 0.0
        if n_unique < 5 or dur < 1.0:
            return fig, f"GNSS time parsing issue: rows={len(gnss)}, unique_t={n_unique}, dur={dur:.3f}s. Check gnss.timestamp format.", None

        # ---- Repair to perfect 10 Hz timeline ----
        repaired, report = elite_gnss_repair(gnss, t_sec_col="_t_sec", v_col="_v", fs=10.0)

        # Store repaired signal for cropping/metrics
        session_payload = repaired[[
            "_t_sec", "v_raw", "v_fill", "v_smooth",
            "is_observed", "fill_method", "confidence"
        ]].to_dict("records")
        
        '''
        # Remove bad rows
        df = df.dropna(subset=["_t", "_v"])

        # Sort chronologically (important!)
        df = df.sort_values("_t").reset_index(drop=True)

        # ---- Convert to seconds from start ----
        t0 = df["_t"].iloc[0]
        df["_t_sec"] = (df["_t"] - t0).dt.total_seconds()
        '''

        # Safety: remove negative or crazy values if API weirdness
        #df = df[df["_t_sec"] >= 0]

        # ---- Plot ----
        obs = repaired[repaired["is_observed"]]
        est = repaired[~repaired["is_observed"] & repaired["v_fill"].notna()]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=obs["_t_sec"], y=obs["v_smooth"],
            mode="markers", marker=dict(size=3),
            name="Observed (smoothed)"
        ))
        fig.add_trace(go.Scattergl(
            x=est["_t_sec"], y=est["v_smooth"],
            mode="markers", marker=dict(size=3),
            name="Estimated (smoothed)"
        ))

        fig.update_layout(
            dragmode="select",
            title=f"GNSS Speed (10 Hz repaired) • Quality {report['quality_score_0_100']:.1f}/100",
            xaxis_title="Time from session start (s)",
            yaxis_title="Speed",
        )


        fig.update_layout(
            dragmode="select",   # enables box select by default
            clickmode="event+select"
        )
        

        #session_payload = df[["_t_sec", "_v"]].to_dict("records")


        return fig, f"Repaired to {len(repaired)} points (10 Hz). Quality {report['quality_score_0_100']:.1f}/100.", session_payload


    except Exception as e:
        return fig, f"Error pulling/plotting: {e}", None
    

@dash.callback(
    Output("gps-crop-store", "data"),
    Output("gps-crop-status", "children"),
    Input("gps-velo-plot", "selectedData"),
    State("gps-session-df-store", "data"),
    prevent_initial_call=True,
)
def gps_crop_from_selection(selectedData, session_data):
    if not session_data:
        return no_update, "No session data loaded."

    if not selectedData or "points" not in selectedData or len(selectedData["points"]) == 0:
        return no_update, "Select a region on the plot to crop."

    # Extract selected x values (time_s)
    xs = [pt.get("x") for pt in selectedData["points"] if pt.get("x") is not None]
    if not xs:
        return no_update, "Selection had no x-values."

    x_min = float(min(xs))
    x_max = float(max(xs))

    # Convert stored data back into arrays
    df = pd.DataFrame(session_data)  # columns: _t_sec, _v
    # Find closest indices to the selected x-range
    # (more robust than exact equality)
    start_idx = int((df["_t_sec"] - x_min).abs().idxmin())
    end_idx = int((df["_t_sec"] - x_max).abs().idxmin())

    section_distance = float(np.nanmean(df["v_fill"].iloc[start_idx:end_idx+1]) * (x_max - x_min))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    crop = {
        "x_min": x_min,
        "x_max": x_max,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }

    return crop, f"Cropped: {x_min:.2f}s → {x_max:.2f}s (idx {start_idx} → {end_idx}), Distance ~ {section_distance:.1f}m"


@dash.callback(
    Output("gps-velo-plot-cropped", "figure"),
    Input("gps-crop-store", "data"),
    State("gps-session-df-store", "data"),
    prevent_initial_call=True,
)
def gps_show_cropped(crop, session_data):
    if not crop or not session_data:
        return go.Figure()

    df = pd.DataFrame(session_data)
    sub = df.iloc[crop["start_idx"]: crop["end_idx"] + 1].copy()
    dt = sub["_t_sec"].diff().fillna(0.1)  # 10 Hz default fallback
    sub["_A"] = sub["v_fill"].diff().fillna(0) / dt


    peaks,_ = find_peaks(lowpass(sub["_A"]*-1,1.2, 10), height = 2, distance = 5)


    stroke_speeds = [0]
    stroke_min_s = [0]
    stroke_max_s = [0]
    stroke_accels = [0]
    stroke_dist = [0]
    stroke_ts = [0]
    stroke_time = [0]


    for i in range(0,len(peaks)): 
        stroke_speed = np.mean(sub["v_fill"].iloc[peaks[i-1]:peaks[i]]) 
        stroke_min_speed = np.min(sub["v_fill"].iloc[peaks[i-1]:peaks[i]])
        stroke_max_speed = np.max(sub["v_fill"].iloc[peaks[i-1]:peaks[i]])
        stroke_ts.append(sub["_t_sec"].iloc[peaks[i-1]])
        stroke_min_s.append(stroke_min_speed)
        stroke_max_s.append(stroke_max_speed)
        stroke_speeds.append(stroke_speed)
        stroke_time.append(sub["_t_sec"].iloc[peaks[i]])
        stroke_accel = np.mean(sub["_A"].iloc[peaks[i-1]:peaks[i]])
        stroke_accels.append(stroke_accel)

        stroke_d = stroke_speed * (sub["_t_sec"].iloc[peaks[i]] - sub["_t_sec"].iloc[peaks[i-1]])
        stroke_dist.append(stroke_d)

    stroke_data = pd.DataFrame()

    stroke_data['speed'] = stroke_speeds
    stroke_data['min speed'] = stroke_min_s
    stroke_data['max speed'] = stroke_max_s
    stroke_data['accel'] = stroke_accels
    #stroke_data['onset_index'] = peaks[1:]
    stroke_data['timestamp'] = stroke_ts
    stroke_data['Prog'] = np.array(stroke_data['speed'])/float(prog_dict["M8+"])*100
    stroke_data['DPS'] = stroke_dist
    stroke_data['Distance'] = np.cumsum(stroke_dist)
    stroke_data['time_s'] = stroke_time 

    
    

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(
        x=sub["_t_sec"],
        y=sub["v_fill"],
        mode="markers",
        marker=dict(size=3),
        name="GNSS speed (cropped)",
    ))
    fig.add_trace(go.Scattergl(
        x=sub["_t_sec"],
        y=sub["_A"],
        mode="lines",
        marker=dict(size=3),
        name="GNSS acceleration (cropped)",
    ), secondary_y=True)

    fig.add_trace(go.Scattergl(
        x=sub["_t_sec"].iloc[peaks],
        y=sub["_A"].iloc[peaks],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Detected peaks",
    ), secondary_y=True)

    fig.update_layout(
        title="Cropped GNSS speed and acceleration",
        xaxis_title="Time (s from start)",
        yaxis_title="Speed",
    )
    return fig


@dash.callback(
    Output("gps-metrics-cards", "children"),
    Input("gps-crop-store", "data"),
    Input("gps-prog-dropdown", "value"),   
    State("gps-session-df-store", "data"),
    prevent_initial_call=True,
)
def gps_build_metric_cards(crop, prog_key, session_data):
    if not crop or not session_data:
        return no_update

    df = pd.DataFrame(session_data)
    sub = df.iloc[crop["start_idx"]: crop["end_idx"] + 1].copy()
    if sub.empty or len(sub) < 10:
        return dbc.Alert("Not enough data in crop to compute metrics.", color="warning")

    # accel
    dt = sub["_t_sec"].diff().fillna(0.1)
    sub["_A"] = sub["v_fill"].diff().fillna(0) / dt

    # peaks = stroke boundaries
    peaks, _ = find_peaks(lowpass(sub["_A"] * -1, 1.2, 10), height=2, distance=5)

    if len(peaks) < 2:
        return dbc.Alert("Not enough strokes detected in this crop (need ≥ 2 peaks).", color="warning")

    # stroke table (skip the first dummy)
    stroke_rows = []
    for i in range(1, len(peaks)):
        i0, i1 = peaks[i - 1], peaks[i]
        if i1 <= i0:
            continue

        seg_v = sub["v_fill"].iloc[i0:i1]
        seg_a = sub["_A"].iloc[i0:i1]
        t0 = float(sub["_t_sec"].iloc[i0])
        t1 = float(sub["_t_sec"].iloc[i1])
        dt_seg = max(1e-6, t1 - t0)

        stroke_speed = float(np.nanmean(seg_v))
        stroke_min_speed = float(np.nanmin(seg_v))
        stroke_max_speed = float(np.nanmax(seg_v))
        stroke_accel = float(np.nanmean(seg_a))
        dps = stroke_speed * dt_seg

        stroke_rows.append(
            {
                "t0": t0,
                "t1": t1,
                "dt": dt_seg,
                "speed": stroke_speed,
                "min_speed": stroke_min_speed,
                "max_speed": stroke_max_speed,
                "accel": stroke_accel,
                "DPS": dps,
            }
        )

    stroke_data = pd.DataFrame(stroke_rows)
    if stroke_data.empty:
        return dbc.Alert("Stroke table came back empty after filtering.", color="warning")
    

    
    # derived metrics
    total_time = float(sub["_t_sec"].iloc[-1] - sub["_t_sec"].iloc[0])
    mean_speed = float(np.nanmean(sub["v_fill"]))
    max_speed = float(np.nanmax(sub["v_fill"]))
    distance = float(np.trapz(sub["v_fill"].values, sub["_t_sec"].values))  # meters

    spm = float(60.0 / np.nanmean(stroke_data["dt"])) if np.nanmean(stroke_data["dt"]) > 0 else np.nan
    mean_dps = float(np.nanmean(stroke_data["DPS"]))
    mean_split = speed_to_split(mean_speed) if mean_speed > 0 else "--:--.---"
    max_split = speed_to_split(max_speed) if max_speed > 0 else "--:--.---"

    # prog (use your dict; guard if key missing)
    ref = float(prog_dict.get(prog_key, np.nan))
    prog_pct = float((mean_speed / ref) * 100.0) if np.isfinite(ref) and ref > 0 else np.nan
    

    cards = dbc.Row(
        [
            dbc.Col(metric_card("Duration", sec_to_split(total_time), f"{total_time:.1f} s"), md=3),
            dbc.Col(metric_card("Distance", f"{distance:.0f} m", "∫speed dt"), md=3),
            dbc.Col(metric_card("Avg Speed", f"{mean_speed:.2f} m/s", f"Split {mean_split} /500"), md=3),
            dbc.Col(metric_card("Max Speed", f"{max_speed:.2f} m/s", f"Split {max_split} /500"), md=3),

            dbc.Col(metric_card("Stroke Rate", f"{spm:.1f} spm", "from peak-to-peak"), md=3),
            dbc.Col(metric_card("Mean DPS", f"{mean_dps:.2f} m", "distance per stroke"), md=3),
            dbc.Col(metric_card("Strokes", f"{len(stroke_data)}", "detected"), md=3),
            dbc.Col(metric_card("Prognostic", f"{prog_pct:.1f}%", "% of Prog"), md=3),
            
        ],
        className="g-3",
    )

    return cards

@dash.callback(
    Output("gps-stroke-store", "data"),
    Output("gps-stroke-metrics-plot", "figure"),
    Input("gps-crop-store", "data"),
    State("gps-session-df-store", "data"),
    State("gps-prog-dropdown", "value"),   # <-- add this if you implement the prog dropdown
    prevent_initial_call=True,
)
def gps_build_stroke_plot(crop, session_data, prog_key):
    if not crop or not session_data:
        return None, go.Figure()

    if not prog_key:
        prog_key = "M8+"

    df = pd.DataFrame(session_data)
    sub = df.iloc[crop["start_idx"]: crop["end_idx"] + 1].copy()

    # acceleration from speed
    dt = sub["_t_sec"].diff().fillna(0.1)  # 10 Hz default fallback
    sub["_A"] = sub["v_fill"].diff().fillna(0) / dt

    # peaks (your existing approach)
    peaks, _ = find_peaks(lowpass(sub["_A"] * -1, 1.2, 10), height=2, distance=5)

    stroke_df = build_stroke_data(sub, peaks, prog_key=prog_key)

    # Make the figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title="Per-stroke Speed & Stroke Rate vs Distance",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (m/s)",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if stroke_df.empty:
        fig.add_annotation(
            text="Not enough peaks detected to compute stroke metrics.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return None, fig

    # Speed vs Distance
    fig.add_trace(
        go.Scattergl(
            x=stroke_df["dist_end"],
            y=stroke_df["speed_mean"],
            mode="lines+markers",
            name="Speed (mean)",
        ),
        secondary_y=False,
    )

    # Stroke rate vs Distance (secondary axis)
    fig.add_trace(
        go.Scattergl(
            x=stroke_df["dist_end"],
            y=stroke_df["stroke_rate_spm"],
            mode="lines+markers",
            name="Stroke rate (spm)",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Speed (m/s)", secondary_y=False)
    fig.update_yaxes(title_text="Stroke rate (spm)", range = [10,50], secondary_y=True)

    return stroke_df.to_dict("records"), fig



@dash.callback(
    Output("gps-download-pdf", "data"),
    Output("gps-pdf-status", "children"),
    Input("gps-btn-download-pdf", "n_clicks"),
    State("gps-session-df-store", "data"),
    State("gps-crop-store", "data"),
    State("gps-seg-m", "value"),
    State("gps-prog-dropdown", "value"),
    prevent_initial_call=True,
)
def gps_download_pdf(n_clicks, session_data, crop, seg_m, prog_key):
    if not n_clicks:
        return no_update, no_update
    if not session_data or not crop:
        return no_update, "No cropped data available yet."

    seg_m = float(seg_m or 100.0)
    prog_key = prog_key or "M8+"

    df = pd.DataFrame(session_data)
    sub = df.iloc[crop["start_idx"]: crop["end_idx"] + 1].copy().reset_index(drop=True)

    sub = sub.reset_index(drop=True)
    dt = sub["_t_sec"].diff().fillna(0.1)
    sub["_A"] = sub["v_fill"].diff().fillna(0) / dt

    peaks, _ = find_peaks(lowpass(sub["_A"] * -1, 1.2, 10), height=2, distance=5)

    # Build per-stroke df with distance columns
    stroke_df = build_stroke_data(sub, peaks, prog_key=prog_key)

    pdf_bytes = build_race_report_pdf_bytes(
        sub=sub,
        stroke_df=stroke_df,
        title="Race Report",
        segment_m=seg_m,
    )
    filename = f"race_report_{int(seg_m)}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return dcc.send_bytes(pdf_bytes, filename), f"Generated {filename} using {int(seg_m)}m segments."