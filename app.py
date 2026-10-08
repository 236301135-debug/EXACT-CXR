# =====================================================================
# PERMANENT SYSTEM & ENVIRONMENT FIXES (MUST BE BEFORE ANY OTHER IMPORTS)
# =====================================================================
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import io
import gc
import json
import time
import base64
import pickle
import datetime
import pathlib
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import torchxrayvision as xrv
from skimage.metrics import structural_similarity as ssim

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.lib.utils import ImageReader

plt.close("all")


# =====================================================================
# STEP 11 MODEL ARTIFACT PATHS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CONFIG_PATH = OUTPUT_DIR / "step11_final_model_config.json"
PNEUMONIA_MODEL_PATH = OUTPUT_DIR / "step11_pneumonia_xrv_feature_lr.pkl"

XRV_WEIGHTS = "densenet121-res224-all"
CARDIOMEGALY_XRV_INDEX = 10
CARDIOMEGALY_THRESHOLD = 0.45
PNEUMONIA_THRESHOLD_DEFAULT = 0.40


# =====================================================================
# LOGO LOADER
# =====================================================================
LOGO_PATH = r"gemini-svg.png"
LOGO_FALLBACK_PATH = LOGO_PATH
_LOGO_CACHE = {"png_bytes": None}


def _load_logo_bytes():
    if _LOGO_CACHE["png_bytes"] is not None:
        return _LOGO_CACHE["png_bytes"]

    path = pathlib.Path(LOGO_PATH)

    if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        data = path.read_bytes()
        _LOGO_CACHE["png_bytes"] = data
        return data

    if path.exists() and path.suffix.lower() == ".svg":
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(url=str(path), output_width=512, output_height=512)
            _LOGO_CACHE["png_bytes"] = png_bytes
            return png_bytes
        except Exception as svg_err:
            fb = pathlib.Path(LOGO_FALLBACK_PATH)
            if fb.exists() and fb.suffix.lower() != ".svg":
                data = fb.read_bytes()
                _LOGO_CACHE["png_bytes"] = data
                return data
            print(f"[LOGO] SVG conversion failed: {svg_err}", file=sys.stderr)

    placeholder = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    _LOGO_CACHE["png_bytes"] = placeholder
    return placeholder


def _logo_bytes():
    return _load_logo_bytes()


def _logo_pil():
    return Image.open(io.BytesIO(_logo_bytes())).convert("RGBA")


def _logo_data_uri():
    b64 = base64.b64encode(_logo_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


LOGO_DATA_URI = _logo_data_uri()


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="EXACT-CXR | Clinical AI Safety Gate",
    page_icon=_logo_pil(),
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CSS (unchanged visual design — kept identical to original)
# =====================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        color: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }

    .boot-screen {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, #ffffff 0%, #eef6ff 100%);
        z-index: 999999;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        animation: bootFade 3.7s ease-in-out forwards;
    }
    @keyframes bootFade {
        0%   { opacity: 1; visibility: visible; }
        85%  { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; display: none; }
    }
    .boot-logo-wrap { position: relative; width: 160px; height: 160px; margin-bottom: 26px; }
    .boot-ring-outer {
        position: absolute; inset: 0; border-radius: 50%;
        border: 3px solid transparent;
        border-top-color: #0ea5e9; border-right-color: #38bdf8;
        animation: spinCW 1.4s linear infinite;
    }
    .boot-ring-inner {
        position: absolute; inset: 16px; border-radius: 50%;
        border: 3px solid transparent;
        border-bottom-color: #0284c7; border-left-color: #7dd3fc;
        animation: spinCCW 1.1s linear infinite;
    }
    .boot-ring-mid {
        position: absolute; inset: 32px; border-radius: 50%;
        border: 2px dashed #bae6fd;
        animation: spinCW 3.2s linear infinite;
    }
    @keyframes spinCW  { from { transform: rotate(0deg); }   to { transform: rotate(360deg); } }
    @keyframes spinCCW { from { transform: rotate(360deg); } to { transform: rotate(0deg); } }
    .boot-logo-core {
        position: absolute; inset: 36px; border-radius: 22px;
        background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 34px rgba(14, 165, 233, 0.55);
        animation: corePulse 1.5s ease-in-out infinite;
        overflow: hidden;
    }
    @keyframes corePulse {
        0%, 100% { transform: scale(1);    box-shadow: 0 0 24px rgba(14,165,233,0.45); }
        50%      { transform: scale(1.06); box-shadow: 0 0 44px rgba(14,165,233,0.75); }
    }
    .boot-logo-core img { width: 72px; height: 72px; object-fit: contain;
        filter: drop-shadow(0 0 6px rgba(14,165,233,0.6)); }
    .boot-title {
        font-size: 1.7rem; font-weight: 800; color: #0f172a;
        letter-spacing: -0.4px; margin-bottom: 6px;
        animation: textFadeIn 0.9s ease-out;
    }
    .boot-title .accent { color: #0ea5e9; }
    .boot-sub {
        font-size: 0.85rem; color: #64748b;
        letter-spacing: 1.2px; text-transform: uppercase;
        font-weight: 600; margin-bottom: 26px;
        animation: textFadeIn 1.2s ease-out;
    }
    @keyframes textFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    .boot-bar-wrap {
        width: 340px; height: 6px; background: #e0f2fe;
        border-radius: 4px; overflow: hidden; margin-bottom: 18px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
    }
    .boot-bar-fill {
        height: 100%; width: 0%;
        background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 50%, #7dd3fc 100%);
        border-radius: 4px;
        animation: bootProgress 3.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    @keyframes bootProgress {
        0%   { width: 0%;   } 15%  { width: 18%;  }
        35%  { width: 42%;  } 60%  { width: 68%;  }
        85%  { width: 92%;  } 100% { width: 100%; }
    }
    .boot-log {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.76rem; color: #475569;
        text-align: left; width: 340px; line-height: 1.8;
    }
    .boot-log .ok { color: #16a34a; font-weight: 700; }
    .boot-log .line { opacity: 0; animation: lineFade 0.35s ease-out forwards; }
    .boot-log .line:nth-child(1) { animation-delay: 0.3s; }
    .boot-log .line:nth-child(2) { animation-delay: 0.9s; }
    .boot-log .line:nth-child(3) { animation-delay: 1.5s; }
    .boot-log .line:nth-child(4) { animation-delay: 2.1s; }
    .boot-log .line:nth-child(5) { animation-delay: 2.7s; }
    @keyframes lineFade {
        from { opacity: 0; transform: translateX(-6px); }
        to   { opacity: 1; transform: translateX(0);    }
    }

    .disclaimer-overlay {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        z-index: 999998;
        display: flex; align-items: center; justify-content: center;
        animation: disclaimerFade 5s ease-in-out forwards;
    }
    @keyframes disclaimerFade {
        0%   { opacity: 0; visibility: visible; }
        6%   { opacity: 1; visibility: visible; }
        85%  { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; display: none; }
    }
    .disclaimer-box {
        position: relative; background: #ffffff;
        border-radius: 18px; padding: 3px;
        max-width: 620px; width: 90%;
        box-shadow: 0 30px 80px rgba(0,0,0,0.45);
        animation: disclaimerPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    @keyframes disclaimerPop {
        from { transform: scale(0.85); opacity: 0; }
        to   { transform: scale(1);    opacity: 1; }
    }
    .disclaimer-box::before {
        content: ""; position: absolute; inset: -3px;
        border-radius: 20px; padding: 3px;
        background: conic-gradient(
            from var(--angle, 0deg),
            #ff3b3b, #ff9a00, #ffee00, #4ade80,
            #22d3ee, #3b82f6, #a855f7, #ff3b3b
        );
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;
                mask-composite: exclude;
        animation: rainbowSpin 3s linear infinite;
        z-index: 0;
    }
    @property --angle {
        syntax: "<angle>";
        initial-value: 0deg;
        inherits: false;
    }
    @keyframes rainbowSpin { to { --angle: 360deg; } }
    .disclaimer-inner {
        position: relative; background: #ffffff;
        border-radius: 16px; padding: 30px 34px 26px 34px; z-index: 1;
    }
    .disclaimer-header {
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 16px; padding-bottom: 14px;
        border-bottom: 1px solid #e2e8f0;
    }
    .disclaimer-header .logo-tile {
        width: 52px; height: 52px; border-radius: 12px;
        background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 12px rgba(14,165,233,0.3);
        flex-shrink: 0; overflow: hidden;
    }
    .disclaimer-header .logo-tile img { width: 40px; height: 40px; object-fit: contain; }
    .disclaimer-header .title-block { flex: 1; min-width: 0; }
    .disclaimer-header .title-1 {
        font-size: 1.15rem; font-weight: 800; color: #0f172a;
        letter-spacing: -0.3px; line-height: 1.2;
    }
    .disclaimer-header .title-2 {
        font-size: 0.78rem; color: #64748b;
        letter-spacing: 0.8px; text-transform: uppercase;
        font-weight: 600; margin-top: 2px;
    }
    .disclaimer-body { font-size: 0.94rem; color: #334155; line-height: 1.65; }
    .disclaimer-body strong { color: #0f172a; }
    .disclaimer-footer {
        margin-top: 18px; padding-top: 14px;
        border-top: 1px solid #e2e8f0;
        display: flex; align-items: center; justify-content: space-between; gap: 12px;
    }
    .disclaimer-footer .countdown-label {
        font-size: 0.78rem; color: #64748b;
        letter-spacing: 0.6px; text-transform: uppercase; font-weight: 600;
    }
    .disclaimer-footer .countdown-bar {
        flex: 1; height: 5px; background: #e2e8f0;
        border-radius: 3px; overflow: hidden;
    }
    .disclaimer-footer .countdown-bar > div {
        height: 100%; width: 100%;
        background: linear-gradient(90deg, #0ea5e9 0%, #a855f7 100%);
        border-radius: 3px;
        animation: countdown 5s linear forwards;
        transform-origin: left center;
    }
    @keyframes countdown {
        from { transform: scaleX(1); }
        to   { transform: scaleX(0); }
    }

    .hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border: 1px solid #dbeafe; border-left: 6px solid #0ea5e9;
        border-radius: 14px; padding: 22px 28px; margin-bottom: 22px;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
        display: flex; align-items: center; gap: 20px;
    }
    .hero-logo-box {
        flex-shrink: 0; width: 76px; height: 76px; border-radius: 16px;
        background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25); overflow: hidden;
    }
    .hero-logo-box img { width: 60px; height: 60px; object-fit: contain; }
    .hero-text-block { flex: 1; min-width: 0; }
    .hero-title {
        font-size: 1.75rem; font-weight: 800; color: #0f172a;
        margin-bottom: 4px; letter-spacing: -0.5px;
        display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    .hero-title .accent { color: #0ea5e9; }
    .hero-subtitle { color: #64748b; font-size: 0.96rem; font-weight: 400; margin: 0; }
    .hero-badge {
        display: inline-block; background: #e0f2fe; color: #0369a1;
        border: 1px solid #bae6fd; padding: 3px 10px; border-radius: 6px;
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.6px; text-transform: uppercase;
    }

    .clinical-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 20px 22px; margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .clinical-card:hover {
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
        border-color: #cbd5e1;
    }
    .decision-accept {
        border-left: 6px solid #16a34a !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%) !important;
    }
    .decision-abstain {
        border-left: 6px solid #dc2626 !important;
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%) !important;
    }
    .status-badge {
        display: inline-block; padding: 5px 14px; border-radius: 6px;
        font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.6px; text-transform: uppercase;
    }
    .badge-pass { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-fail { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

    .finding-title {
        font-size: 1.75rem; font-weight: 800; color: #0f172a;
        margin: 14px 0 6px 0; letter-spacing: -0.4px;
    }
    .finding-title .highlight { color: #0284c7; }
    .finding-meta { color: #64748b; font-size: 0.98rem; margin: 0; }
    .finding-meta strong { color: #0f172a; font-weight: 700; }

    .ess-score-block { text-align: right; min-width: 130px; }
    .ess-value { font-size: 2.1rem; font-weight: 800; line-height: 1; }
    .ess-label {
        font-size: 0.72rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.8px; margin-top: 4px; font-weight: 600;
    }
    .audit-note {
        background: #f8fafc; border-left: 3px solid #94a3b8;
        padding: 12px 16px; border-radius: 6px; margin-top: 16px;
        font-size: 0.92rem; color: #334155; line-height: 1.6;
    }
    .audit-note strong { color: #0f172a; }

    .metric-label {
        font-size: 0.72rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.7px; font-weight: 600; margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem; font-weight: 800; color: #0f172a; line-height: 1.15;
    }
    .metric-sub { font-size: 0.78rem; color: #64748b; margin-top: 3px; }

    .path-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 16px 14px; text-align: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease;
    }
    .path-card.top {
        border-color: #0ea5e9;
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.12);
    }
    .path-card .p-name {
        font-size: 0.78rem; color: #475569; text-transform: uppercase;
        letter-spacing: 0.5px; font-weight: 700; margin-bottom: 8px;
    }
    .path-card .p-value { font-size: 1.6rem; font-weight: 800; color: #0f172a; line-height: 1; }
    .path-card.top .p-value { color: #0284c7; }
    .path-card .p-logit { font-size: 0.72rem; color: #94a3b8; margin-top: 6px; }

    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #0f172a;
        margin: 24px 0 14px 0; padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        display: flex; align-items: center; gap: 10px;
    }
    .section-header::before {
        content: ""; display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; background: #0ea5e9;
        box-shadow: 0 0 0 3px #e0f2fe; flex-shrink: 0;
    }
    .card-head { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
    .card-head::before {
        content: ""; display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; background: #0ea5e9;
        box-shadow: 0 0 0 3px #e0f2fe; flex-shrink: 0;
    }
    .card-head .card-title { font-weight: 700; color: #0f172a; font-size: 0.95rem; }

    div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] * { color: #0f172a; }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stSelectbox label {
        color: #334155 !important; font-weight: 600 !important; font-size: 0.9rem !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        background: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0ea5e9; background: #f0f9ff;
    }
    .stImage figcaption {
        text-align: center !important;
        font-size: 0.8rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    hr { border-color: #e2e8f0 !important; }
    div[data-testid="stAlert"] { border-radius: 10px; }

    .footer-note {
        text-align: center; color: #94a3b8; font-size: 0.8rem;
        margin-top: 30px; padding: 16px; border-top: 1px solid #e2e8f0;
    }
    .side-section {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.85rem; font-weight: 700; color: #0284c7;
        text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
    }
    .side-section::before {
        content: ""; display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: #0ea5e9; flex-shrink: 0;
    }
    .panel-label {
        text-align: center; color: #334155; font-weight: 700;
        font-size: 0.88rem; margin-bottom: 10px; padding: 6px;
        background: #f1f5f9; border-radius: 6px;
        display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .panel-label::before {
        content: ""; display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: #0ea5e9; flex-shrink: 0;
    }
    .feature-list {
        margin-top: 22px; display: flex; justify-content: center;
        gap: 22px; flex-wrap: wrap; font-size: 0.82rem; color: #475569;
    }
    .feature-list .feat { display: flex; align-items: center; gap: 6px; }
    .feature-list .feat::before {
        content: ""; display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: #16a34a; flex-shrink: 0;
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.4px !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stDownloadButton > button:hover {
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    .sidebar-logo-head {
        display: flex; align-items: center; gap: 12px;
        padding: 4px 0 16px 0; border-bottom: 2px solid #e2e8f0;
        margin-bottom: 18px;
    }
    .sidebar-logo-head .logo-box {
        width: 44px; height: 44px; border-radius: 10px;
        background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.2);
        overflow: hidden; flex-shrink: 0;
    }
    .sidebar-logo-head .logo-box img { width: 34px; height: 34px; object-fit: contain; }
    .sidebar-logo-head .text-block { min-width: 0; }
    .sidebar-logo-head .t1 {
        font-size: 1.05rem; font-weight: 800; color: #0f172a; line-height: 1.15;
    }
    .sidebar-logo-head .t2 {
        font-size: 0.72rem; color: #64748b; margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# BOOT ANIMATION
# =====================================================================
if "boot_complete" not in st.session_state:
    boot_placeholder = st.empty()
    boot_placeholder.markdown(f"""
    <div class="boot-screen">
        <div class="boot-logo-wrap">
            <div class="boot-ring-outer"></div>
            <div class="boot-ring-inner"></div>
            <div class="boot-ring-mid"></div>
            <div class="boot-logo-core">
                <img src="{LOGO_DATA_URI}" alt="EXACT-CXR Logo"/>
            </div>
        </div>
        <div class="boot-title">EXACT-<span class="accent">CXR</span></div>
        <div class="boot-sub">Clinical Safety Engine · Booting</div>
        <div class="boot-bar-wrap">
            <div class="boot-bar-fill"></div>
        </div>
        <div class="boot-log">
            <div class="line"><span class="ok">[OK]</span> Initializing runtime environment</div>
            <div class="line"><span class="ok">[OK]</span> Loading TorchXRayVision DenseNet-121</div>
            <div class="line"><span class="ok">[OK]</span> Loading pneumonia LR artifact</div>
            <div class="line"><span class="ok">[OK]</span> Registering Grad-CAM hooks</div>
            <div class="line"><span class="ok">[OK]</span> Safety gate armed — ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3.8)
    boot_placeholder.empty()
    st.session_state["boot_complete"] = True


# =====================================================================
# STEP 11 ARTIFACT VALIDATION (HARD FAIL)
# =====================================================================
_missing = []
if not CONFIG_PATH.exists():
    _missing.append(str(CONFIG_PATH))
if not PNEUMONIA_MODEL_PATH.exists():
    _missing.append(str(PNEUMONIA_MODEL_PATH))

if _missing:
    st.error("### ⚠️ Required Step-11 model artifacts are missing")
    for p in _missing:
        st.code(p)
    st.info(
        "Place `step11_final_model_config.json` and "
        "`step11_pneumonia_xrv_feature_lr.pkl` inside the `outputs/` "
        "folder next to this script, then restart."
    )
    st.stop()


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_pneumonia_artifact():
    with open(PNEUMONIA_MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    config = load_config()
except Exception as exc:
    st.error("Unable to load Step 11 configuration.")
    st.exception(exc)
    st.stop()


def validate_pneumonia_artifact(artifact):
    if not isinstance(artifact, dict):
        raise ValueError("Pneumonia artifact is not a dictionary.")
    for key in ("scaler", "classifier"):
        if key not in artifact:
            raise ValueError(f"Pneumonia artifact missing key: {key}")

    scaler = artifact["scaler"]
    classifier = artifact["classifier"]

    if not hasattr(scaler, "transform"):
        raise ValueError("Pneumonia scaler is invalid.")
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Pneumonia classifier lacks predict_proba().")

    feature_dim = None
    if hasattr(scaler, "n_features_in_"):
        feature_dim = int(scaler.n_features_in_)
    elif hasattr(classifier, "n_features_in_"):
        feature_dim = int(classifier.n_features_in_)
    elif hasattr(classifier, "coef_"):
        feature_dim = int(classifier.coef_.shape[1])

    if feature_dim is None:
        raise ValueError("Unable to determine pneumonia feature dimension.")
    if feature_dim != 1024:
        raise ValueError(
            f"Unexpected pneumonia feature dim: {feature_dim}. Expected 1024."
        )

    threshold = float(artifact.get("threshold", PNEUMONIA_THRESHOLD_DEFAULT))
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Invalid pneumonia threshold: {threshold}")

    return {
        "feature_dim": feature_dim,
        "threshold": threshold,
        "scaler_type": type(scaler).__name__,
        "classifier_type": type(classifier).__name__,
    }


try:
    pneumonia_artifact = load_pneumonia_artifact()
    pneumonia_info = validate_pneumonia_artifact(pneumonia_artifact)
except Exception as exc:
    st.error("Pneumonia model artifact validation failed.")
    st.exception(exc)
    st.stop()

PNEUMONIA_THRESHOLD = float(pneumonia_info["threshold"])


# =====================================================================
# TORCHXRAYVISION MODEL
# =====================================================================
DEVICE = torch.device("cpu")


@st.cache_resource
def load_xrv_model():
    m = xrv.models.DenseNet(weights=XRV_WEIGHTS)
    m = m.to(DEVICE)
    m.eval()
    return m


with st.spinner("Initializing Clinical Inference Engine..."):
    xrv_model = load_xrv_model()


# =====================================================================
# GRAD-CAM WRAPPER FOR CARDIOMEGALY (TorchXRayVision's DenseNet)
# =====================================================================
class CardioGradCAM:
    """
    Attaches forward / backward hooks to the XRV DenseNet-121
    `features.denseblock4` module, mirroring the original EXACT-CXR hook
    strategy but now wired to the real TorchXRayVision backbone.
    """

    def __init__(self, xrv_model):
        self.model = xrv_model
        self.activations = None
        self.gradients = None
        self._handles = []
        target = self.model.features.denseblock4
        self._handles.append(target.register_forward_hook(self._fwd))
        self._handles.append(target.register_full_backward_hook(self._bwd))

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def clear(self):
        self.activations = None
        self.gradients = None

    def get_cam(self):
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
        grads = self.gradients.detach().cpu().numpy()[0]
        fmap = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i, :, :]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (224, 224))


gradcam = CardioGradCAM(xrv_model)


# =====================================================================
# PREPROCESSING (Step-11 exact)
# =====================================================================
def preprocess_xray_step11(uploaded_file):
    """
    uint16 image
        → [0, 65535]
        → [-1024, 1024]
        → XRayCenterCrop
        → XRayResizer(224)
        → torch.float32 tensor [1,224,224]
    """
    uploaded_file.seek(0)

    # Preserve 16-bit where possible; PIL "I" gives 32-bit int for 16-bit sources
    pil_img = Image.open(uploaded_file)

    # For display we keep the original PIL
    if pil_img.mode in ("I", "I;16", "I;16B", "I;16L"):
        arr = np.asarray(pil_img, dtype=np.float32)
    else:
        # 8-bit sources (PNG/JPG): scale to 16-bit-like range
        arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
        arr = arr * (65535.0 / 255.0)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Image contains non-finite values.")

    arr = ((arr / 65535.0) * 2048.0) - 1024.0
    arr = arr[None, ...]  # [1,H,W]

    arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)

    if arr.shape != (1, 224, 224):
        raise ValueError(f"Unexpected preprocessed shape: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Preprocessed image contains non-finite values.")

    tensor = torch.from_numpy(arr).float().to(DEVICE)
    return pil_img, arr, tensor


def make_display_image(pil_img):
    """
    Display-only 8-bit conversion. Does NOT affect inference.
    """
    arr = np.asarray(pil_img, dtype=np.float32)
    if arr.ndim == 2:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("No valid finite pixel values for display.")
        low = float(np.percentile(finite, 1))
        high = float(np.percentile(finite, 99))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high > low:
            arr = (arr - low) / (high - low)
        else:
            arr = np.zeros_like(arr)
        arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    return pil_img.convert("RGB")


# =====================================================================
# INFERENCE
# =====================================================================
@torch.no_grad()
def extract_xrv_features(image_tensor):
    batch = image_tensor.unsqueeze(0)
    feats = xrv_model.features(batch)
    if feats.ndim == 4:
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(start_dim=1)
    elif feats.ndim != 2:
        raise ValueError(f"Unexpected XRV feature shape: {tuple(feats.shape)}")
    if feats.shape[1] != 1024:
        raise ValueError(f"Expected 1024 features; got {feats.shape[1]}")
    return feats.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_cardiomegaly_prob(image_tensor):
    batch = image_tensor.unsqueeze(0)
    out = xrv_model(batch)
    if isinstance(out, dict):
        out = out["logits"]
    val = float(out[0, CARDIOMEGALY_XRV_INDEX].item())
    if val < 0.0 or val > 1.0:
        val = float(torch.sigmoid(out[0, CARDIOMEGALY_XRV_INDEX]).item())
    return val


def predict_pneumonia_prob(feature_vector):
    scaler = pneumonia_artifact["scaler"]
    clf = pneumonia_artifact["classifier"]
    scaled = scaler.transform(feature_vector)
    probs = clf.predict_proba(scaled)
    if probs.shape[1] != 2:
        raise ValueError(
            f"Expected binary pneumonia classifier; got {probs.shape[1]} columns."
        )
    return float(probs[0, 1])


def cardio_gradcam(image_tensor):
    """Grad-CAM for cardiomegaly using real XRV backward pass."""
    gradcam.clear()
    inp = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    xrv_model.zero_grad(set_to_none=True)
    out = xrv_model(inp)
    if isinstance(out, dict):
        out = out["logits"]
    score = out[0, CARDIOMEGALY_XRV_INDEX]
    score.backward()
    cam = gradcam.get_cam()
    xrv_model.zero_grad(set_to_none=True)
    gradcam.clear()
    gc.collect()
    return cam


def pneumonia_saliency(image_tensor):
    """
    Saliency map for the pneumonia pipeline.

    The pneumonia classifier is a *sklearn* LogisticRegression applied to
    frozen 1024-D XRV features, so there is no end-to-end differentiable
    path all the way to the pixels. We therefore compute an input-gradient
    saliency of the L2-norm of the feature vector, which highlights the
    spatial regions that most strongly drive the XRV feature extractor.

    This is documented in the UI so users understand the difference
    between cardiomegaly Grad-CAM and pneumonia saliency.
    """
    inp = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    xrv_model.zero_grad(set_to_none=True)
    feats = xrv_model.features(inp)
    if feats.ndim == 4:
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(start_dim=1)
        score = pooled.pow(2).sum()
        grads = torch.autograd.grad(score, inp, retain_graph=False)[0]
        sal = grads.abs().squeeze().detach().cpu().numpy()
    else:
        sal = np.zeros((224, 224), dtype=np.float32)
    xrv_model.zero_grad(set_to_none=True)
    gc.collect()
    if sal.max() > 0:
        sal = sal / sal.max()
    return cv2.resize(sal, (224, 224))


# =====================================================================
# SAFETY ENGINE HELPERS
# =====================================================================
def evaluate_quality_gate(pil_img):
    w, h = pil_img.size
    gray = np.array(pil_img.convert("L"))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_std = float(np.std(gray))
    mean_intensity = float(np.mean(gray))
    score = float(
        np.clip(
            (contrast_std / 70.0) * 0.5 + (min(laplacian_var, 200.0) / 200.0) * 0.5,
            0.1, 0.99,
        )
    )
    is_valid = (w >= 224 and h >= 224) and contrast_std >= 20.0 and laplacian_var >= 15.0
    return is_valid, score, laplacian_var, contrast_std, mean_intensity, (w, h)


def calculate_ess(base_cam, p_cam):
    b = base_cam.flatten()
    p = p_cam.flatten()
    cos_sim = float(np.dot(b, p) / ((np.linalg.norm(b) * np.linalg.norm(p)) + 1e-8))
    ssim_val = float(ssim(base_cam, p_cam, data_range=1.0))
    ess = 0.5 * cos_sim + 0.5 * ssim_val
    return ess, cos_sim, ssim_val


def blend_heatmap(img_hwc, cam):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.65 * img_hwc + 0.35 * heat, 0, 1)


def apply_input_perturbation(tensor_img, mode="noise", intensity=1.0):
    """
    Works on the [-1024, 1024] normalized tensors used by XRV.
    Perturbation scale is set relative to the dynamic range (2048).
    """
    with torch.no_grad():
        p = tensor_img.clone()
        if mode in ("Gaussian Noise", "All Combined"):
            p = p + torch.randn_like(p) * (0.02 * intensity * 2048.0 / 10.0)
        if mode in ("Micro Rotation", "All Combined"):
            p = transforms.functional.rotate(p, int(3 * intensity))
        if mode in ("Contrast Shift", "All Combined"):
            # multiplicative contrast around the tensor mean
            m = p.mean()
            p = (p - m) * (1.0 + 0.15 * intensity) + m
    return p


def numpy_to_pil_bytes(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    buf.seek(0)
    return buf


# =====================================================================
# PDF REPORT GENERATION (unchanged design, updated content)
# =====================================================================
BRAND_BLUE = colors.HexColor("#0ea5e9")
BRAND_BLUE_DARK = colors.HexColor("#0369a1")
BRAND_SLATE_DARK = colors.HexColor("#0f172a")
BRAND_SLATE_MID = colors.HexColor("#64748b")
BRAND_SLATE_LIGHT = colors.HexColor("#e2e8f0")
BRAND_BG_LIGHT = colors.HexColor("#f8fafc")
WATERMARK_ALPHA = 0.08
WATERMARK_COLOR = colors.HexColor("#94a3b8")


def _draw_watermark_only(canvas_obj, doc):
    try:
        w, h = A4
        canvas_obj.saveState()
        logo_reader = ImageReader(io.BytesIO(_logo_bytes()))
        canvas_obj.translate(w / 2, h / 2)
        canvas_obj.rotate(25)
        canvas_obj.setFillAlpha(WATERMARK_ALPHA)
        canvas_obj.drawImage(
            logo_reader, -65 * mm, -65 * mm,
            width=130 * mm, height=130 * mm, mask="auto",
        )
        canvas_obj.setFillColor(WATERMARK_COLOR)
        canvas_obj.setFillAlpha(WATERMARK_ALPHA * 0.55)
        canvas_obj.setFont("Helvetica-Bold", 46)
        canvas_obj.drawCentredString(0, -100 * mm, "EXACT-CXR")
        canvas_obj.setFont("Helvetica-Bold", 11)
        canvas_obj.drawCentredString(0, -108 * mm, "CLINICAL SAFETY ENGINE")
        canvas_obj.restoreState()
    except Exception:
        pass


def _draw_header_footer(canvas_obj, doc):
    try:
        w, h = A4
        canvas_obj.saveState()
        canvas_obj.setFillColor(BRAND_BLUE)
        canvas_obj.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        try:
            header_logo = ImageReader(io.BytesIO(_logo_bytes()))
            canvas_obj.drawImage(
                header_logo, 8 * mm, h - 12.5 * mm,
                width=9 * mm, height=9 * mm, mask="auto",
            )
        except Exception:
            pass
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont("Helvetica-Bold", 11)
        canvas_obj.drawString(20 * mm, h - 9.5 * mm, "EXACT-CXR  |  Clinical Safety Report")
        canvas_obj.setFont("Helvetica", 8.5)
        canvas_obj.drawRightString(
            w - 15 * mm, h - 9.5 * mm,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        canvas_obj.restoreState()

        canvas_obj.saveState()
        canvas_obj.setStrokeColor(BRAND_SLATE_LIGHT)
        canvas_obj.setLineWidth(0.6)
        canvas_obj.line(15 * mm, 14 * mm, w - 15 * mm, 14 * mm)
        canvas_obj.setFillColor(BRAND_SLATE_MID)
        canvas_obj.setFont("Helvetica", 7.5)
        canvas_obj.drawString(
            15 * mm, 9.5 * mm,
            "Research Use Only · Not for primary diagnostic decision-making · EXACT-CXR v7.3",
        )
        canvas_obj.setFont("Helvetica-Bold", 8)
        canvas_obj.setFillColor(BRAND_BLUE_DARK)
        canvas_obj.drawRightString(
            w - 15 * mm, 9.5 * mm,
            f"Page {canvas_obj.getPageNumber()}",
        )
        canvas_obj.restoreState()
    except Exception:
        pass


def _on_first_page(c, d):
    _draw_watermark_only(c, d)
    _draw_header_footer(c, d)


def _on_later_pages(c, d):
    _draw_watermark_only(c, d)
    _draw_header_footer(c, d)


def _make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleX", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=BRAND_SLATE_DARK,
            spaceAfter=2, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "SubTitleX", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=13, textColor=BRAND_SLATE_MID, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2X", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=BRAND_BLUE_DARK,
            spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyX", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=BRAND_SLATE_DARK,
        ),
        "body_j": ParagraphStyle(
            "BodyJ", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=BRAND_SLATE_DARK,
            alignment=4,  # justify
        ),
        "caption": ParagraphStyle(
            "CaptionX", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8, leading=11, textColor=BRAND_SLATE_MID, alignment=TA_CENTER,
        ),
    }


def _kv_table(rows, col_widths, header_bg=BRAND_BLUE):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), BRAND_SLATE_DARK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, BRAND_SLATE_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def generate_pdf_report(
    input_image_pil, base_cam, p_cam, diff_map,
    predicted_disease, top_prob, calibrated_conf, raw_logits, probs, top_idx,
    ess_score, cos_sim, ssim_val, is_accepted, rejection_reasons,
    qual_score, lap_var, contrast_std, mean_intensity, img_dims, is_qual_valid,
    temp_scaling, pert_type, pert_intensity, t_qual, t_conf, t_ess,
    cardio_prob, pneumo_prob,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="EXACT-CXR Clinical Safety Report",
        author="EXACT-CXR Clinical Safety Engine",
        subject="Chest X-Ray AI Safety Report",
        onFirstPage=_on_first_page,
        onLaterPages=_on_later_pages,
    )
    styles = _make_styles()
    story = []

    story.append(Paragraph("EXACT-CXR — Clinical Safety Report", styles["title"]))
    story.append(Paragraph(
        "Explanation-Stable, Confidence-Aware &amp; Cross-Domain Chest X-Ray Diagnostic Gate "
        "(TorchXRayVision DenseNet-121 + Pneumonia LR)",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_SLATE_LIGHT, spaceAfter=10))

    meta_rows = [
        ["Report ID", f"EXCXR-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"],
        ["Generated At", datetime.datetime.now().strftime("%A, %d %B %Y · %H:%M:%S")],
        ["Application", "EXACT-CXR Clinical Safety Engine v7.3"],
        ["Model Backbone", "TorchXRayVision DenseNet-121 (densenet121-res224-all)"],
        ["Pneumonia Head", "Frozen XRV features (1024-D) → StandardScaler → LogisticRegression"],
        ["Analysis Mode", f"{pert_type} · Intensity {pert_intensity:.1f}× · T={temp_scaling:.2f}"],
        ["Intended Use", "Research use only. Not for primary diagnostic decision-making."],
    ]
    story.append(_kv_table(meta_rows, [45 * mm, 133 * mm]))
    story.append(Spacer(1, 12))

    # Executive decision
    story.append(Paragraph("1. Executive Decision", styles["h2"]))
    decision_text = "PASSED — Prediction Accepted" if is_accepted else "DEFERRED — Abstained for Expert Review"
    decision_color = colors.HexColor("#15803d") if is_accepted else colors.HexColor("#b91c1c")
    decision_bg = colors.HexColor("#f0fdf4") if is_accepted else colors.HexColor("#fef2f2")

    decision_tbl = Table(
        [[Paragraph(
            f"<font color='{decision_color.hexval()}'><b>STATUS:</b> {decision_text}</font><br/><br/>"
            f"<b>Primary Finding:</b> <font color='#0284c7'><b>{predicted_disease}</b></font><br/>"
            f"<b>Cardiomegaly:</b> {cardio_prob:.1%} &nbsp;|&nbsp; "
            f"<b>Pneumonia:</b> {pneumo_prob:.1%}<br/>"
            f"<b>Calibrated Probability:</b> {top_prob:.1%} &nbsp;|&nbsp; "
            f"<b>Logit (top):</b> {raw_logits[top_idx]:.3f} &nbsp;|&nbsp; "
            f"<b>Confidence:</b> {calibrated_conf:.1%}<br/>"
            f"<b>Explanation Stability (ESS):</b> {ess_score:.3f}",
            styles["body"],
        )]],
        colWidths=[178 * mm], hAlign="LEFT",
    )
    decision_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_bg),
        ("BOX", (0, 0), (-1, -1), 1.2, decision_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(decision_tbl)
    story.append(Spacer(1, 8))

    if is_accepted:
        audit = (
            "All safety thresholds were met. Visual saliency maps demonstrate high spatial "
            f"stability (ESS = {ess_score:.2f}) under {pert_type} perturbation, confirming the "
            "model relies on robust anatomical indicators rather than artifact shortcuts."
        )
    else:
        audit = (
            f"Safety gate triggered deferral to radiologist. Reason(s): "
            f"{'; '.join(rejection_reasons) if rejection_reasons else 'Unspecified'}. "
            "Visual explanation shifts indicate model sensitivity to input noise or "
            "out-of-distribution features."
        )
    story.append(Paragraph("<b>Clinical &amp; Safety Audit Note:</b>", styles["body"]))
    story.append(Paragraph(audit, styles["body_j"]))
    story.append(Spacer(1, 10))

    # Input + matrix
    story.append(Paragraph("2. Input Radiograph &amp; Diagnostic Matrix", styles["h2"]))
    input_img_resized = input_image_pil.convert("RGB").resize((300, 300))
    input_buf = io.BytesIO()
    input_img_resized.save(input_buf, format="PNG")
    input_buf.seek(0)
    rl_input_img = RLImage(input_buf, width=62 * mm, height=62 * mm)

    path_rows = [["Pathology", "Prob.", "Logit", "Top"]]
    for i, disease in enumerate(PATHOLOGIES):
        path_rows.append([disease, f"{probs[i]:.1%}", f"{raw_logits[i]:.3f}", "★" if i == top_idx else ""])

    path_tbl = Table(path_rows, colWidths=[40 * mm, 26 * mm, 24 * mm, 16 * mm], hAlign="LEFT")
    path_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, BRAND_SLATE_LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    for i in range(len(PATHOLOGIES)):
        if i == top_idx:
            path_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#e0f2fe")),
                ("TEXTCOLOR", (0, i + 1), (-1, i + 1), BRAND_BLUE_DARK),
                ("FONTNAME", (0, i + 1), (-1, i + 1), "Helvetica-Bold"),
            ]))

    combined = Table(
        [[rl_input_img, "", path_tbl]],
        colWidths=[62 * mm, 10 * mm, 108 * mm], hAlign="LEFT",
    )
    combined.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(combined)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Figure 1 — Uploaded chest radiograph (left) and multi-pathology probability matrix (right).",
        styles["caption"],
    ))
    story.append(Spacer(1, 10))

    # Audit boards
    story.append(Paragraph("3. System Parameter Audit Board", styles["h2"]))
    quality_rows = [
        ["Parameter", "Value"],
        ["Image Dimensions", f"{img_dims[0]} × {img_dims[1]} px"],
        ["Quality Score", f"{qual_score:.3f}"],
        ["Sharpness (Laplacian Var.)", f"{lap_var:.2f}"],
        ["Contrast Std. Dev.", f"{contrast_std:.2f}"],
        ["Mean Intensity", f"{mean_intensity:.2f}"],
        ["Quality Gate Valid", "YES" if is_qual_valid else "NO"],
    ]
    calibration_rows = [
        ["Parameter", "Value"],
        ["Temperature (T)", f"{temp_scaling:.2f}"],
        ["Cardiomegaly Prob.", f"{cardio_prob:.3f}"],
        ["Pneumonia Prob.", f"{pneumo_prob:.3f}"],
        ["Top Class Probability", f"{top_prob:.1%}"],
    ]
    stability_rows = [
        ["Parameter", "Value"],
        ["Cosine Similarity", f"{cos_sim:.4f}"],
        ["Structural SSIM", f"{ssim_val:.4f}"],
        ["Composite ESS", f"{ess_score:.4f}"],
        ["ESS Threshold", f"{t_ess:.2f}"],
        ["ESS Status", "PASS" if ess_score >= t_ess else "FAIL"],
    ]
    tbl_q = _kv_table(quality_rows, [42 * mm, 16 * mm])
    tbl_c = _kv_table(calibration_rows, [42 * mm, 16 * mm], header_bg=colors.HexColor("#6366f1"))
    tbl_s = _kv_table(stability_rows, [42 * mm, 16 * mm], header_bg=colors.HexColor("#8b5cf6"))

    audit_grid = Table([[tbl_q, tbl_c, tbl_s]],
                       colWidths=[59 * mm, 59 * mm, 59 * mm], hAlign="LEFT")
    audit_grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(audit_grid)
    story.append(Spacer(1, 12))

    # Visual panels
    story.append(Paragraph("4. Explanation Stability &amp; Visual Attributions", styles["h2"]))
    base_rgb = (blend_heatmap(
        cv2.resize(np.array(input_image_pil.convert("RGB")), (224, 224)) / 255.0, base_cam
    ) * 255).astype(np.uint8)
    pert_rgb = (blend_heatmap(
        cv2.resize(np.array(input_image_pil.convert("RGB")), (224, 224)) / 255.0, p_cam
    ) * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=140)
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    im = ax.imshow(diff_map, cmap="inferno")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors="#475569")
    plt.axis("off"); plt.tight_layout()
    diff_buf = io.BytesIO()
    plt.savefig(diff_buf, format="png", bbox_inches="tight", dpi=140, facecolor="#ffffff")
    plt.close(fig); diff_buf.seek(0)

    base_buf = numpy_to_pil_bytes(base_rgb)
    pert_buf = numpy_to_pil_bytes(pert_rgb)

    img_w = img_h = 50 * mm
    rl_base = RLImage(base_buf, width=img_w, height=img_h)
    rl_pert = RLImage(pert_buf, width=img_w, height=img_h)
    rl_diff = RLImage(diff_buf, width=img_w, height=img_h)

    vis_rows = [
        [rl_base, "", rl_pert, "", rl_diff],
        [
            Paragraph("<b>Baseline Attribution</b><br/>Original Saliency", styles["caption"]), "",
            Paragraph(f"<b>Attribution under {pert_type}</b><br/>{pert_intensity:.1f}× Scale", styles["caption"]), "",
            Paragraph(f"<b>Attribution Variance Map</b><br/>ESS = {ess_score:.3f}", styles["caption"]),
        ],
    ]
    vis_tbl = Table(vis_rows, colWidths=[54 * mm, 3 * mm, 54 * mm, 3 * mm, 54 * mm], hAlign="CENTER")
    vis_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(vis_tbl)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Figure 2 — Saliency maps: baseline attribution (left), attribution after perturbation (center), "
        "and spatial variance between the two (right). Lower variance indicates higher explanation stability.",
        styles["caption"],
    ))
    story.append(Spacer(1, 12))

    # Config
    story.append(Paragraph("5. Configuration Summary", styles["h2"]))
    config_rows = [
        ["Setting", "Value"],
        ["Min Image Quality Threshold (T_qual)", f"{t_qual:.2f}"],
        ["Min Calibrated Confidence (T_conf)", f"{t_conf:.1%}"],
        ["Min Explanation Stability (T_ess)", f"{t_ess:.2f}"],
        ["Temperature Scaling (T)", f"{temp_scaling:.2f}"],
        ["Perturbation Mode", pert_type],
        ["Perturbation Intensity", f"{pert_intensity:.1f}×"],
        ["Cardiomegaly Threshold", f"{CARDIOMEGALY_THRESHOLD:.2f}"],
        ["Pneumonia Threshold", f"{PNEUMONIA_THRESHOLD:.2f}"],
        ["Overall Gate Result", "PASS" if is_accepted else "ABSTAIN"],
    ]
    story.append(_kv_table(config_rows, [95 * mm, 83 * mm]))
    story.append(Spacer(1, 10))

    # Gate details
    story.append(Paragraph("6. Safety Gate Evaluation Details", styles["h2"]))
    gate_checks = [
        ["Gate Criterion", "Measured", "Threshold", "Result"],
        ["Image Quality", f"{qual_score:.3f}", f"≥ {t_qual:.2f}",
         "PASS" if (qual_score >= t_qual and is_qual_valid) else "FAIL"],
        ["Calibrated Confidence", f"{calibrated_conf:.1%}", f"≥ {t_conf:.1%}",
         "PASS" if calibrated_conf >= t_conf else "FAIL"],
        ["Explanation Stability (ESS)", f"{ess_score:.3f}", f"≥ {t_ess:.2f}",
         "PASS" if ess_score >= t_ess else "FAIL"],
    ]
    gate_tbl = Table(gate_checks, colWidths=[72 * mm, 36 * mm, 36 * mm, 34 * mm], hAlign="LEFT")
    gate_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, BRAND_SLATE_LIGHT),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    for i in range(1, len(gate_checks)):
        color_ = colors.HexColor("#15803d") if gate_checks[i][3] == "PASS" else colors.HexColor("#b91c1c")
        gate_tbl.setStyle(TableStyle([
            ("TEXTCOLOR", (3, i), (3, i), color_),
            ("FONTNAME", (3, i), (3, i), "Helvetica-Bold"),
        ]))
    story.append(gate_tbl)
    story.append(Spacer(1, 12))

    story.append(HRFlowable(width="100%", thickness=0.6, color=BRAND_SLATE_LIGHT, spaceAfter=8))
    story.append(Paragraph("<b>Disclaimer</b>", styles["body"]))
    story.append(Paragraph(
        "This report is generated by the EXACT-CXR Clinical Safety Engine for research and "
        "educational purposes only. It is not a medical device and must not be used as the sole "
        "basis for clinical diagnosis, treatment, or patient management decisions. All findings "
        "must be reviewed and confirmed by a qualified radiologist or licensed clinician. "
        "EXACT-CXR and its authors disclaim any liability for clinical decisions made based on "
        "this output.",
        styles["body_j"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo-head">
        <div class="logo-box">
            <img src="{LOGO_DATA_URI}" alt="logo"/>
        </div>
        <div class="text-block">
            <div class="t1">Control Panel</div>
            <div class="t2">Configure safety gate parameters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='side-section'>Safety Thresholds</div>", unsafe_allow_html=True)
    t_qual = st.slider("Min Image Quality (T_qual)", 0.0, 1.0, 0.45, 0.05)
    t_conf = st.slider("Min Calibrated Confidence (T_conf)", 0.50, 1.0, 0.70, 0.05)
    t_ess = st.slider("Min Explanation Stability (T_ess)", 0.0, 1.0, 0.65, 0.05)

    st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='side-section'>Model & Perturbation</div>", unsafe_allow_html=True)
    temp_scaling = st.slider("Temperature Scaling (T)", 0.5, 3.0, 1.35, 0.05)
    pert_type = st.selectbox(
        "Perturbation Engine Mode",
        ["Gaussian Noise", "Micro Rotation", "Contrast Shift", "All Combined"],
    )
    pert_intensity = st.slider("Perturbation Intensity", 0.1, 3.0, 1.0, 0.1)

    st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='side-section'>Model Artifacts</div>", unsafe_allow_html=True)
    with st.expander("Inspect loaded models", expanded=False):
        st.write("**XRV weights:**", XRV_WEIGHTS)
        st.write("**Cardiomegaly XRV index:**", CARDIOMEGALY_XRV_INDEX)
        st.write("**Cardiomegaly threshold:**", CARDIOMEGALY_THRESHOLD)
        st.write("**Pneumonia feature dim:**", pneumonia_info["feature_dim"])
        st.write("**Pneumonia scaler:**", pneumonia_info["scaler_type"])
        st.write("**Pneumonia classifier:**", pneumonia_info["classifier_type"])
        st.write("**Pneumonia threshold:**", PNEUMONIA_THRESHOLD)
        if config:
            st.json(config, expanded=False)

    st.markdown("""
    <div style="margin-top: 24px; padding: 12px 14px; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; font-size: 0.78rem; color: #075985; line-height: 1.5;">
        <strong>Clinical Note:</strong> All thresholds are calibrated for research use. Adjust per institutional protocol.
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# DISCLAIMER OVERLAY
# =====================================================================
def show_disclaimer_overlay(duration_seconds=5):
    overlay_html = f"""
    <div class="disclaimer-overlay">
        <div class="disclaimer-box">
            <div class="disclaimer-inner">
                <div class="disclaimer-header">
                    <div class="logo-tile">
                        <img src="{LOGO_DATA_URI}" alt="logo"/>
                    </div>
                    <div class="title-block">
                        <div class="title-1">Clinical Use Disclaimer</div>
                        <div class="title-2">Please read before reviewing results</div>
                    </div>
                </div>
                <div class="disclaimer-body">
                    This research tool is <strong>NOT a certified medical device</strong> and is not cleared for clinical diagnosis.
                    All generated findings, scores, and saliency maps are for <strong>informational purposes only</strong> and must be reviewed and confirmed by a qualified clinician before making any diagnostic decisions.
                </div>
                <div class="disclaimer-footer">
                    <div class="countdown-label">Results loading…</div>
                    <div class="countdown-bar"><div></div></div>
                </div>
            </div>
        </div>
    </div>
    """
    placeholder = st.empty()
    placeholder.markdown(overlay_html, unsafe_allow_html=True)
    time.sleep(duration_seconds)
    placeholder.empty()


# =====================================================================
# PATHOLOGIES (only the two real ones)
# =====================================================================
PATHOLOGIES = ["Cardiomegaly", "Pneumonia"]


# =====================================================================
# HERO + UPLOADER
# =====================================================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-logo-box">
        <img src="{LOGO_DATA_URI}" alt="EXACT-CXR logo"/>
    </div>
    <div class="hero-text-block">
        <div class="hero-title">
            <span>EXACT-CXR</span> <span class="accent">Clinical Safety Engine</span>
            <span class="hero-badge">v7.3</span>
        </div>
        <p class="hero-subtitle">Explanation-Stable, Confidence-Aware &amp; Cross-Domain Chest X-Ray Diagnostic Gate (TorchXRayVision DenseNet-121 + Pneumonia LR)</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Research-use tool · Not a certified medical device · "
    "Full clinical disclaimer will be presented after you upload a radiograph."
)

uploaded_file = st.file_uploader(
    "Upload Frontal Chest Radiograph (PNG / JPG / JPEG / TIF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Supported: PNG, JPG, JPEG, TIF, TIFF (16-bit preferred). Minimum 224×224 px.",
)


# =====================================================================
# MAIN FLOW
# =====================================================================
if uploaded_file is not None:
    try:
        with st.spinner("Executing Quality Gate, TorchXRayVision Inference & Attribution Stability Analysis..."):
            # Preprocess
            original_pil, processed_array, image_tensor = preprocess_xray_step11(uploaded_file)
            display_pil = make_display_image(original_pil)

            # Quality gate
            (
                is_qual_valid, qual_score, lap_var, contrast_std, mean_intensity, img_dims
            ) = evaluate_quality_gate(display_pil)

            # Real models
            cardio_prob = predict_cardiomegaly_prob(image_tensor)
            xrv_feats = extract_xrv_features(image_tensor)
            pneumo_prob = predict_pneumonia_prob(xrv_feats)

            # Build a "logit-like" score for the display
            # (logit of the probability for reporting consistency)
            def _logit(p):
                p = min(max(p, 1e-6), 1 - 1e-6)
                return float(np.log(p / (1 - p)))

            raw_logits = np.array([_logit(cardio_prob), _logit(pneumo_prob)], dtype=np.float32)
            calibrated_logits = raw_logits / temp_scaling
            probs = 1.0 / (1.0 + np.exp(-calibrated_logits))

            top_idx = int(np.argmax(probs))
            predicted_disease = PATHOLOGIES[top_idx]
            top_prob = float(probs[top_idx])
            calibrated_conf = float(max(top_prob, 1.0 - top_prob))

            # --- Attribution stability ---
            # Primary saliency = cardiomegaly Grad-CAM (real differentiable path).
            # Pneumonia saliency shown as supplementary panel.
            base_cam = cardio_gradcam(image_tensor)

            # Perturb & recompute
            pert_tensor = apply_input_perturbation(image_tensor, pert_type, pert_intensity)
            p_cam = cardio_gradcam(pert_tensor)
            ess_score, cos_sim, ssim_val = calculate_ess(base_cam, p_cam)

            # Pneumonia saliency (supplementary, not used in ESS)
            pneumo_saliency = pneumonia_saliency(image_tensor)

            # Safety gate
            rejection_reasons = []
            if qual_score < t_qual or not is_qual_valid:
                rejection_reasons.append(
                    f"Image Quality ({qual_score:.2f}) below threshold ({t_qual:.2f})"
                )
            if calibrated_conf < t_conf:
                rejection_reasons.append(
                    f"Calibrated Confidence ({calibrated_conf:.1%}) below threshold ({t_conf:.1%})"
                )
            if ess_score < t_ess:
                rejection_reasons.append(
                    f"Explanation Stability ESS ({ess_score:.2f}) below threshold ({t_ess:.2f})"
                )
            is_accepted = len(rejection_reasons) == 0

        # Show disclaimer once per unique upload
        _run_id = f"{uploaded_file.name}_{uploaded_file.size}"
        _disclaimer_key = f"disclaimer_shown_for_{_run_id}"
        if not st.session_state.get(_disclaimer_key, False):
            show_disclaimer_overlay(duration_seconds=5)
            st.session_state[_disclaimer_key] = True

        if st.session_state.get("toast_fired_for") != _run_id:
            st.toast("Analysis complete — results ready below.", icon="✅")
            st.session_state["toast_fired_for"] = _run_id

        # --- Input image display ---
        st.markdown('<div class="section-header">Input Radiograph</div>', unsafe_allow_html=True)
        ci1, ci2 = st.columns([1, 2])
        with ci1:
            st.image(display_pil, caption=uploaded_file.name, use_container_width=True)
        with ci2:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-head"><span class="card-title">Image Metadata</span></div>
                <div style="margin-top: 8px;"><div class="metric-label">Original Size</div>
                <div class="metric-value">{img_dims[0]}×{img_dims[1]}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">PIL Mode (original)</div>
                <div class="metric-value" style="font-size: 1.05rem;">{original_pil.mode}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Preprocessed Tensor</div>
                <div class="metric-sub">{tuple(image_tensor.shape)} · {image_tensor.dtype}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Tensor Range</div>
                <div class="metric-sub">[{float(image_tensor.min()):.2f}, {float(image_tensor.max()):.2f}]</div></div>
            </div>
            """, unsafe_allow_html=True)

        # --- Decision card ---
        card_class = "decision-accept" if is_accepted else "decision-abstain"
        badge_class = "badge-pass" if is_accepted else "badge-fail"
        status_text = "PASSED — PREDICTION ACCEPTED" if is_accepted else "DEFERRED — ABSTAINED FOR EXPERT REVIEW"
        ess_color = "#16a34a" if is_accepted else "#dc2626"

        st.markdown(f"""
        <div class="clinical-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px;">
                    <span class="status-badge {badge_class}">{status_text}</span>
                    <div class="finding-title">
                        Primary Finding: <span class="highlight">{predicted_disease}</span>
                    </div>
                    <p class="finding-meta">
                        Calibrated Probability: <strong>{top_prob:.1%}</strong> &nbsp;·&nbsp;
                        Logit Score: <strong>{raw_logits[top_idx]:.2f}</strong> &nbsp;·&nbsp;
                        Confidence: <strong>{calibrated_conf:.1%}</strong>
                    </p>
                </div>
                <div class="ess-score-block">
                    <div class="ess-value" style="color: {ess_color};">{ess_score:.2f}</div>
                    <div class="ess-label">ESS Stability Score</div>
                </div>
            </div>
            <div class="audit-note">
                <strong>Clinical &amp; Safety Audit Note:</strong><br>
                {(
                    f"All safety thresholds met. Visual saliency maps demonstrate high spatial stability (ESS = {ess_score:.2f}) under <em>{pert_type}</em> perturbation, confirming the model relies on robust anatomical indicators rather than artifact shortcuts."
                    if is_accepted else
                    f"Safety gate triggered deferral to radiologist. Reason(s): {', '.join(rejection_reasons)}. Visual explanation shifts indicate model sensitivity to input noise or out-of-distribution features."
                )}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Diagnostic matrix (only 2 real classes now) ---
        st.markdown('<div class="section-header">Diagnostic Matrix (Real Step-11 Models)</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            top_class = "top" if top_idx == 0 else ""
            st.markdown(f"""
            <div class="path-card {top_class}">
                <div class="p-name">Cardiomegaly</div>
                <div class="p-value">{probs[0]:.1%}</div>
                <div class="p-logit">Logit: {raw_logits[0]:.2f} · Threshold: {CARDIOMEGALY_THRESHOLD:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            top_class = "top" if top_idx == 1 else ""
            st.markdown(f"""
            <div class="path-card {top_class}">
                <div class="p-name">Pneumonia</div>
                <div class="p-value">{probs[1]:.1%}</div>
                <div class="p-logit">Logit: {raw_logits[1]:.2f} · Threshold: {PNEUMONIA_THRESHOLD:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Audit board ---
        st.markdown('<div class="section-header">System Parameter Audit Board</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-head"><span class="card-title">Image Quality Gate</span></div>
                <div style="margin-top: 10px;"><div class="metric-label">Dimensions</div>
                <div class="metric-value">{img_dims[0]}×{img_dims[1]} <span style="font-size:0.75rem; font-weight:500; color:#64748b;">px</span></div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Sharpness (Laplacian)</div>
                <div class="metric-value">{lap_var:.1f}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Contrast Std Dev</div>
                <div class="metric-value">{contrast_std:.1f}</div></div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-head"><span class="card-title">Calibration Layer</span></div>
                <div style="margin-top: 10px;"><div class="metric-label">Temperature (T)</div>
                <div class="metric-value">{temp_scaling:.2f}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Cardiomegaly Prob</div>
                <div class="metric-value">{probs[0]:.1%}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Pneumonia Prob</div>
                <div class="metric-value">{probs[1]:.1%}</div></div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            ess_color_panel = "#16a34a" if ess_score >= t_ess else "#dc2626"
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-head"><span class="card-title">Stability Engine</span></div>
                <div style="margin-top: 10px;"><div class="metric-label">Cosine Similarity</div>
                <div class="metric-value">{cos_sim:.3f}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Structural SSIM</div>
                <div class="metric-value">{ssim_val:.3f}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Composite ESS Score</div>
                <div class="metric-value" style="color: {ess_color_panel};">{ess_score:.3f}</div></div>
            </div>
            """, unsafe_allow_html=True)

        with p4:
            gate_color = "#16a34a" if is_accepted else "#dc2626"
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-head"><span class="card-title">Config Summary</span></div>
                <div style="margin-top: 10px;"><div class="metric-label">Gate Thresholds</div>
                <div class="metric-sub">Qual: <strong>{t_qual:.2f}</strong> · Conf: <strong>{t_conf:.0%}</strong> · ESS: <strong>{t_ess:.2f}</strong></div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Perturbation Setup</div>
                <div class="metric-sub"><strong>{pert_type}</strong> ({pert_intensity:.1f}×)</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Overall Gate Result</div>
                <div class="metric-value" style="color: {gate_color}; font-size: 1.1rem; font-weight: 800;">
                    {"PASS" if is_accepted else "ABSTAIN"}</div></div>
            </div>
            """, unsafe_allow_html=True)

        # --- Visual attributions (3 panels) ---
        st.markdown('<div class="section-header">Explanation Stability &amp; Visual Attributions</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)

        # For display we resize the ORIGINAL PIL to 224×224 RGB
        img_resized = cv2.resize(np.array(display_pil.convert("RGB")), (224, 224)) / 255.0

        with v1:
            st.markdown("<div class='panel-label'>Baseline Attribution (Grad-CAM)</div>", unsafe_allow_html=True)
            st.image(
                blend_heatmap(img_resized, base_cam),
                use_container_width=True,
                caption=f"Cardiomegaly Grad-CAM — Top: {predicted_disease}",
            )
        with v2:
            st.markdown(f"<div class='panel-label'>Attribution under {pert_type}</div>", unsafe_allow_html=True)
            st.image(
                blend_heatmap(img_resized, p_cam),
                use_container_width=True,
                caption=f"Perturbed Saliency — {pert_intensity}× Scale",
            )
        with v3:
            st.markdown("<div class='panel-label'>Attribution Variance Map</div>", unsafe_allow_html=True)
            diff_map = np.abs(base_cam - p_cam)
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
            im = ax.imshow(diff_map, cmap="inferno")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8, colors="#475569")
            plt.axis("off"); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
            st.markdown(
                f"<div style='text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 4px;'>"
                f"Spatial Instability Map · ESS = <strong>{ess_score:.3f}</strong></div>",
                unsafe_allow_html=True,
            )

        # Supplementary pneumonia saliency (informational)
        with st.expander("Supplementary: Pneumonia feature saliency (input-gradient on XRV features)", expanded=False):
            st.markdown(
                "The pneumonia classifier is a **sklearn LogisticRegression** operating on frozen "
                "1024-D XRV features. Because there is no end-to-end differentiable path from pixels "
                "to the LR logit, we display an **input-gradient saliency** of the XRV feature L2-norm. "
                "This indicates which spatial regions most strongly drive the XRV feature extractor, "
                "and is *not* an end-to-end pneumonia explanation."
            )
            st.image(
                blend_heatmap(img_resized, pneumo_saliency),
                use_container_width=True,
                caption=f"Pneumonia pipeline saliency · Probability = {probs[1]:.1%}",
            )

        st.warning(
            "**Clinical Caution:** The findings above are generated by a research model and "
            "**must be confirmed by a qualified radiologist**. Do not act on these results "
            "without expert human review. Use of this tool does not replace clinical judgment."
        )

        # --- PDF export ---
        st.markdown('<div class="section-header">Export Clinical Report</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="clinical-card" style="background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%); border-left: 6px solid #0ea5e9;">
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="width:52px; height:52px; border-radius:12px; background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%); display:flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(14,165,233,0.25);">
                    <img src="{LOGO_DATA_URI}" style="width:40px;height:40px;object-fit:contain;" alt="logo"/>
                </div>
                <div style="flex:1;">
                    <div style="font-weight:800; color:#0f172a; font-size:0.98rem;">Generate &amp; Download Full PDF Report</div>
                    <p style="color: #475569; font-size: 0.88rem; line-height: 1.55; margin: 4px 0 0 0;">
                        Full clinical report with the input radiograph, executive decision, real
                        cardiomegaly + pneumonia probabilities, system audit boards, all three visual
                        attribution maps, safety gate breakdown, configuration summary, and disclaimer.
                        Soft-layered with the EXACT-CXR logo watermark on every page.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            pdf_buffer = generate_pdf_report(
                input_image_pil=display_pil,
                base_cam=base_cam, p_cam=p_cam, diff_map=np.abs(base_cam - p_cam),
                predicted_disease=predicted_disease, top_prob=float(top_prob),
                calibrated_conf=float(calibrated_conf), raw_logits=raw_logits,
                probs=probs, top_idx=int(top_idx),
                ess_score=float(ess_score), cos_sim=float(cos_sim), ssim_val=float(ssim_val),
                is_accepted=bool(is_accepted), rejection_reasons=rejection_reasons,
                qual_score=float(qual_score), lap_var=float(lap_var),
                contrast_std=float(contrast_std), mean_intensity=float(mean_intensity),
                img_dims=tuple(img_dims), is_qual_valid=bool(is_qual_valid),
                temp_scaling=float(temp_scaling), pert_type=pert_type,
                pert_intensity=float(pert_intensity),
                t_qual=float(t_qual), t_conf=float(t_conf), t_ess=float(t_ess),
                cardio_prob=float(cardio_prob), pneumo_prob=float(pneumo_prob),
            )
            report_filename = f"EXACT_CXR_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="Download Full PDF Clinical Report",
                data=pdf_buffer,
                file_name=report_filename,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.error(f"PDF Report Generation Error: {str(pdf_err)}")
            st.info("The interactive dashboard is unaffected. Please retry or check reportlab installation.")

        st.markdown("""
        <div class="footer-note">
            EXACT-CXR Clinical Safety Engine · Research Use Only · Not for primary diagnostic decision-making
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        st.info("Please verify the uploaded image format and try again.")

else:
    st.markdown(f"""
    <div class="clinical-card" style="text-align: center; padding: 48px 32px; background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%); border: 2px dashed #bae6fd;">
        <div style="display:inline-flex; align-items:center; justify-content:center; width:80px; height:80px; border-radius:20px; background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%); box-shadow:0 4px 14px rgba(14,165,233,0.25); margin-bottom:16px;">
            <img src="{LOGO_DATA_URI}" style="width:60px;height:60px;object-fit:contain;" alt="logo"/>
        </div>
        <div style="font-size: 1.3rem; font-weight: 800; color: #0284c7; margin-bottom: 10px; letter-spacing: 0.5px;">
            ● WAITING FOR RADIOGRAPH INPUT
        </div>
        <p style="color: #64748b; margin: 0 auto; max-width: 520px; font-size: 0.95rem; line-height: 1.6;">
            Upload a frontal chest X-ray image above to run real multi-pathology inference
            (TorchXRayVision DenseNet-121 for Cardiomegaly + Linear-LR for Pneumonia),
            quality evaluation, calibration analysis, and attribution stability checks.
        </p>
        <div class="feature-list">
            <span class="feat">Real XRV Cardiomegaly</span>
            <span class="feat">Real Pneumonia LR</span>
            <span class="feat">Temperature Calibration</span>
            <span class="feat">Grad-CAM Stability (ESS)</span>
            <span class="feat">Quality Gating</span>
            <span class="feat">PDF Report Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
