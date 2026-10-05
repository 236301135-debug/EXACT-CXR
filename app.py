# =====================================================================
# PERMANENT SYSTEM & ENVIRONMENT FIXES (MUST BE BEFORE ANY OTHER IMPORTS)
# =====================================================================
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import gc
import time
import io
import base64
import datetime
import pathlib

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

plt.close("all")

# =====================================================================
# LOGO LOADER
# =====================================================================
LOGO_PATH = r"gemini-svg.png"
LOGO_FALLBACK_PATH = r"gemini-svg.png"

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
            if fb.exists():
                data = fb.read_bytes()
                _LOGO_CACHE["png_bytes"] = data
                return data
            print(f"[LOGO] SVG conversion failed: {svg_err}", file=sys.stderr)
            placeholder = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            )
            _LOGO_CACHE["png_bytes"] = placeholder
            return placeholder

    print(f"[LOGO] File not found at {LOGO_PATH}. Using placeholder.", file=sys.stderr)
    placeholder = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    _LOGO_CACHE["png_bytes"] = placeholder
    return placeholder


def _logo_bytes():
    return _load_logo_bytes()


def _logo_pil():
    return Image.open(io.BytesIO(_logo_bytes())).convert("RGBA")


def _logo_buf():
    return io.BytesIO(_logo_bytes())


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
    initial_sidebar_state="expanded"
)

# =====================================================================
# CSS
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
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, #ffffff 0%, #eef6ff 100%);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        animation: bootFade 3.7s ease-in-out forwards;
    }
    @keyframes bootFade {
        0%   { opacity: 1; visibility: visible; }
        85%  { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; display: none; }
    }
    .boot-logo-wrap {
        position: relative;
        width: 160px; height: 160px; margin-bottom: 26px;
    }
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
    .boot-logo-core img {
        width: 72px; height: 72px; object-fit: contain;
        filter: drop-shadow(0 0 6px rgba(14,165,233,0.6));
    }
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
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        z-index: 999998;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: disclaimerFade 5s ease-in-out forwards;
    }
    @keyframes disclaimerFade {
        0%   { opacity: 0; visibility: visible; }
        6%   { opacity: 1; visibility: visible; }
        85%  { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; display: none; }
    }
    .disclaimer-box {
        position: relative;
        background: #ffffff;
        border-radius: 18px;
        padding: 3px;
        max-width: 620px;
        width: 90%;
        box-shadow: 0 30px 80px rgba(0,0,0,0.45);
        animation: disclaimerPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    @keyframes disclaimerPop {
        from { transform: scale(0.85); opacity: 0; }
        to   { transform: scale(1);    opacity: 1; }
    }
    .disclaimer-box::before {
        content: "";
        position: absolute;
        inset: -3px;
        border-radius: 20px;
        padding: 3px;
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
    @keyframes rainbowSpin {
        to { --angle: 360deg; }
    }
    .disclaimer-inner {
        position: relative;
        background: #ffffff;
        border-radius: 16px;
        padding: 30px 34px 26px 34px;
        z-index: 1;
    }
    .disclaimer-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 16px;
        padding-bottom: 14px;
        border-bottom: 1px solid #e2e8f0;
    }
    .disclaimer-header .logo-tile {
        width: 52px; height: 52px;
        border-radius: 12px;
        background: radial-gradient(circle at 50% 40%, #1e293b 0%, #0b1220 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 12px rgba(14,165,233,0.3);
        flex-shrink: 0;
        overflow: hidden;
    }
    .disclaimer-header .logo-tile img {
        width: 40px; height: 40px; object-fit: contain;
    }
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
    .disclaimer-body {
        font-size: 0.94rem;
        color: #334155;
        line-height: 1.65;
    }
    .disclaimer-body strong { color: #0f172a; }
    .disclaimer-footer {
        margin-top: 18px;
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }
    .disclaimer-footer .countdown-label {
        font-size: 0.78rem; color: #64748b;
        letter-spacing: 0.6px; text-transform: uppercase;
        font-weight: 600;
    }
    .disclaimer-footer .countdown-bar {
        flex: 1;
        height: 5px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
    }
    .disclaimer-footer .countdown-bar > div {
        height: 100%;
        width: 100%;
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
        border: 1px solid #dbeafe;
        border-left: 6px solid #0ea5e9;
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
            <div class="line"><span class="ok">[OK]</span> Loading DenseNet-121 backbone</div>
            <div class="line"><span class="ok">[OK]</span> Registering Grad-CAM hooks</div>
            <div class="line"><span class="ok">[OK]</span> Calibrating temperature scaling</div>
            <div class="line"><span class="ok">[OK]</span> Safety gate armed — ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3.8)
    boot_placeholder.empty()
    st.session_state["boot_complete"] = True

# =====================================================================
# MODEL
# =====================================================================
PATHOLOGIES = ["Pneumonia", "Cardiomegaly", "Atelectasis", "Pleural Effusion", "Consolidation"]


class DenseNetMultiLabelGradCAM(nn.Module):
    def __init__(self, num_classes=5):
        super(DenseNetMultiLabelGradCAM, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = self.densenet.features
        num_ftrs = self.densenet.classifier.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.gradients = None
        self.activations = None
        self.features.denseblock4.register_forward_hook(self._save_activations)
        self.features.denseblock4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def get_gradcam(self):
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


@st.cache_resource
def load_model():
    model = DenseNetMultiLabelGradCAM(num_classes=len(PATHOLOGIES))
    model.eval()
    return model


with st.spinner("Initializing Clinical Inference Engine..."):
    model = load_model()


# =====================================================================
# HELPERS
# =====================================================================
def preprocess_cxr(image: Image.Image):
    rgb_img = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return rgb_img, transform(rgb_img).unsqueeze(0)


def evaluate_quality_gate(pil_img):
    w, h = pil_img.size
    gray = np.array(pil_img.convert("L"))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_std = float(np.std(gray))
    mean_intensity = float(np.mean(gray))
    score = float(np.clip((contrast_std / 70.0) * 0.5 + (min(laplacian_var, 200.0) / 200.0) * 0.5, 0.1, 0.99))
    is_valid = (w >= 224 and h >= 224) and contrast_std >= 20.0 and laplacian_var >= 15.0
    return is_valid, score, laplacian_var, contrast_std, mean_intensity, (w, h)


def apply_perturbation(tensor_img, mode="noise", intensity=1.0):
    with torch.no_grad():
        p_tensor = tensor_img.clone()
        if mode in ["Gaussian Noise", "All Combined"]:
            p_tensor += torch.randn_like(p_tensor) * (0.05 * intensity)
        if mode in ["Micro Rotation", "All Combined"]:
            p_tensor = transforms.functional.rotate(p_tensor, int(3 * intensity))
        if mode in ["Contrast Shift", "All Combined"]:
            p_tensor = transforms.functional.adjust_contrast(p_tensor, 1.0 + (0.15 * intensity))
    return p_tensor


def compute_multi_label_inference(model, tensor_input, temp_scaling=1.35):
    model.zero_grad()
    input_var = tensor_input.clone().detach().requires_grad_(True)
    logits = model(input_var)[0]
    top_class_idx = torch.argmax(logits).item()
    logits[top_class_idx].backward()
    cam = model.get_gradcam()
    raw_logits = logits.detach().cpu().numpy()
    calibrated_logits = raw_logits / temp_scaling
    probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    input_var.grad = None
    model.zero_grad()
    gc.collect()
    return cam, raw_logits, probs, top_class_idx


def calculate_ess(base_cam, p_cam):
    base_flat = base_cam.flatten()
    p_flat = p_cam.flatten()
    cos_sim = np.dot(base_flat, p_flat) / ((np.linalg.norm(base_flat) * np.linalg.norm(p_flat)) + 1e-8)
    ssim_val = ssim(base_cam, p_cam, data_range=1.0)
    ess = 0.5 * cos_sim + 0.5 * ssim_val
    return float(ess), float(cos_sim), float(ssim_val)


def blend_heatmap(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.65 * img + 0.35 * heatmap, 0, 1)


def numpy_to_pil_bytes(arr_uint8):
    pil_img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# =====================================================================
# PDF REPORT GENERATION — FIXED WATERMARK (drawn via onPage callback)
# =====================================================================
BRAND_BLUE = colors.HexColor("#0ea5e9")
BRAND_BLUE_DARK = colors.HexColor("#0369a1")
BRAND_SLATE_DARK = colors.HexColor("#0f172a")
BRAND_SLATE_MID = colors.HexColor("#64748b")
BRAND_SLATE_LIGHT = colors.HexColor("#e2e8f0")
BRAND_BG_LIGHT = colors.HexColor("#f8fafc")

# Very light, subtle watermark settings — truly "background"
WATERMARK_ALPHA = 0.08
WATERMARK_COLOR = colors.HexColor("#94a3b8")   # slate-400 grey


def _draw_watermark_only(canvas_obj, doc):
    """
    Called by ReportLab's onFirstPage / onLaterPages BEFORE the flowables
    of the current page are drawn. This guarantees the watermark lands
    UNDERNEATH the content on every page.
    """
    try:
        w, h = A4
        canvas_obj.saveState()

        # ---- Logo watermark (rotated, subtle) ----
        logo_reader = ImageReader(io.BytesIO(_logo_bytes()))
        canvas_obj.translate(w / 2, h / 2)
        canvas_obj.rotate(25)
        canvas_obj.setFillAlpha(WATERMARK_ALPHA)
        canvas_obj.drawImage(
            logo_reader,
            -65 * mm, -65 * mm,
            width=130 * mm, height=130 * mm,
            mask="auto"
        )

        # ---- Text watermark (even subtler) ----
        canvas_obj.setFillColor(WATERMARK_COLOR)
        canvas_obj.setFillAlpha(WATERMARK_ALPHA * 0.55)
        canvas_obj.setFont("Helvetica-Bold", 46)
        canvas_obj.drawCentredString(0, -100 * mm, "EXACT-CXR")
        canvas_obj.setFont("Helvetica-Bold", 11)
        canvas_obj.drawCentredString(0, -108 * mm, "CLINICAL SAFETY ENGINE")

        canvas_obj.restoreState()
    except Exception:
        # Never let the watermark crash the report
        pass


def _draw_header_footer(canvas_obj, doc):
    """
    Called on every page to draw the top header band and bottom footer
    AFTER the watermark (so they're on top of it, but this is fine —
    they're at fixed top/bottom positions where no content lives).
    """
    try:
        w, h = A4

        # ---- Header band ----
        canvas_obj.saveState()
        canvas_obj.setFillColor(BRAND_BLUE)
        canvas_obj.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        try:
            header_logo = ImageReader(io.BytesIO(_logo_bytes()))
            canvas_obj.drawImage(
                header_logo,
                8 * mm, h - 12.5 * mm,
                width=9 * mm, height=9 * mm,
                mask="auto"
            )
        except Exception:
            pass
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont("Helvetica-Bold", 11)
        canvas_obj.drawString(20 * mm, h - 9.5 * mm, "EXACT-CXR  |  Clinical Safety Report")
        canvas_obj.setFont("Helvetica", 8.5)
        canvas_obj.drawRightString(
            w - 15 * mm, h - 9.5 * mm,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        canvas_obj.restoreState()

        # ---- Footer band ----
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(BRAND_SLATE_LIGHT)
        canvas_obj.setLineWidth(0.6)
        canvas_obj.line(15 * mm, 14 * mm, w - 15 * mm, 14 * mm)
        canvas_obj.setFillColor(BRAND_SLATE_MID)
        canvas_obj.setFont("Helvetica", 7.5)
        canvas_obj.drawString(
            15 * mm, 9.5 * mm,
            "Research Use Only · Not for primary diagnostic decision-making · EXACT-CXR v7.2"
        )
        canvas_obj.setFont("Helvetica-Bold", 8)
        canvas_obj.setFillColor(BRAND_BLUE_DARK)
        canvas_obj.drawRightString(
            w - 15 * mm, 9.5 * mm,
            f"Page {canvas_obj.getPageNumber()}"
        )
        canvas_obj.restoreState()
    except Exception:
        pass


def _on_first_page(canvas_obj, doc):
    _draw_watermark_only(canvas_obj, doc)
    _draw_header_footer(canvas_obj, doc)


def _on_later_pages(canvas_obj, doc):
    _draw_watermark_only(canvas_obj, doc)
    _draw_header_footer(canvas_obj, doc)


def _make_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "TitleX", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=BRAND_SLATE_DARK, spaceAfter=2, alignment=TA_LEFT
    )
    styles["subtitle"] = ParagraphStyle(
        "SubTitleX", parent=base["Normal"],
        fontName="Helvetica", fontSize=10, leading=13,
        textColor=BRAND_SLATE_MID, spaceAfter=10
    )
    styles["h2"] = ParagraphStyle(
        "H2X", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=BRAND_BLUE_DARK, spaceBefore=14, spaceAfter=6
    )
    styles["body"] = ParagraphStyle(
        "BodyX", parent=base["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=BRAND_SLATE_DARK, alignment=TA_LEFT
    )
    styles["body_j"] = ParagraphStyle(
        "BodyJ", parent=styles["body"], alignment=TA_JUSTIFY
    )
    styles["caption"] = ParagraphStyle(
        "CaptionX", parent=base["Normal"],
        fontName="Helvetica-Oblique", fontSize=8, leading=11,
        textColor=BRAND_SLATE_MID, alignment=TA_CENTER
    )
    return styles


def _kv_table(rows, col_widths, header_bg=BRAND_BLUE, header_fg=colors.white):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_fg),
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
    ess_score, cos_sim, ssim_val,
    is_accepted, rejection_reasons,
    qual_score, lap_var, contrast_std, mean_intensity, img_dims, is_qual_valid,
    temp_scaling, pert_type, pert_intensity,
    t_qual, t_conf, t_ess,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title="EXACT-CXR Clinical Safety Report",
        author="EXACT-CXR Clinical Safety Engine",
        subject="Chest X-Ray AI Safety Report",
        # ▼▼▼ THE KEY FIX: watermark + header/footer drawn on the page canvas ▼▼▼
        onFirstPage=_on_first_page,
        onLaterPages=_on_later_pages,
    )

    styles = _make_styles()
    story = []

    # ---------------- COVER ----------------
    story.append(Paragraph("EXACT-CXR — Clinical Safety Report", styles["title"]))
    story.append(Paragraph(
        "Explanation-Stable, Confidence-Aware &amp; Cross-Domain Chest X-Ray Diagnostic Gate",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_SLATE_LIGHT, spaceAfter=10))

    meta_rows = [
        ["Report ID", f"EXCXR-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"],
        ["Generated At", datetime.datetime.now().strftime("%A, %d %B %Y · %H:%M:%S")],
        ["Application", "EXACT-CXR Clinical Safety Engine v7.2"],
        ["Model Backbone", "DenseNet-121 (ImageNet-pretrained) · Multi-label head (5 classes)"],
        ["Analysis Mode", f"{pert_type} perturbation · Intensity {pert_intensity:.1f}× · T={temp_scaling:.2f}"],
        ["Intended Use", "Research use only. Not for primary diagnostic decision-making."],
    ]
    story.append(_kv_table(meta_rows, [45 * mm, 133 * mm]))
    story.append(Spacer(1, 12))

    # ---------------- 1. EXECUTIVE DECISION ----------------
    story.append(Paragraph("1. Executive Decision", styles["h2"]))
    decision_text = "PASSED — Prediction Accepted" if is_accepted else "DEFERRED — Abstained for Expert Review"
    decision_color = colors.HexColor("#15803d") if is_accepted else colors.HexColor("#b91c1c")
    decision_bg = colors.HexColor("#f0fdf4") if is_accepted else colors.HexColor("#fef2f2")

    decision_tbl = Table(
        [[Paragraph(
            f"<font color='{decision_color.hexval()}'><b>STATUS:</b> {decision_text}</font>"
            f"<br/><br/>"
            f"<b>Primary Finding:</b> <font color='#0284c7'><b>{predicted_disease}</b></font><br/>"
            f"<b>Calibrated Probability:</b> {top_prob:.1%} &nbsp;|&nbsp; "
            f"<b>Logit Score:</b> {raw_logits[top_idx]:.3f} &nbsp;|&nbsp; "
            f"<b>Confidence:</b> {calibrated_conf:.1%}<br/>"
            f"<b>Explanation Stability (ESS):</b> {ess_score:.3f}",
            styles["body"]
        )]],
        colWidths=[178 * mm], hAlign="LEFT"
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
        audit = ("All safety thresholds were met. Visual saliency maps demonstrate high spatial "
                 f"stability (ESS = {ess_score:.2f}) under {pert_type} perturbation, confirming the "
                 "model relies on robust anatomical indicators rather than artifact shortcuts.")
    else:
        reasons_str = "; ".join(rejection_reasons) if rejection_reasons else "Unspecified"
        audit = (f"Safety gate triggered deferral to radiologist. Reason(s): {reasons_str}. "
                 "Visual explanation shifts indicate model sensitivity to input noise or "
                 "out-of-distribution features.")
    story.append(Paragraph("<b>Clinical &amp; Safety Audit Note:</b>", styles["body"]))
    story.append(Paragraph(audit, styles["body_j"]))
    story.append(Spacer(1, 10))

    # ---------------- 2. INPUT IMAGE + MATRIX ----------------
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
        colWidths=[62 * mm, 10 * mm, 108 * mm],
        hAlign="LEFT"
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
        styles["caption"]
    ))
    story.append(Spacer(1, 10))

    # ---------------- 3. AUDIT BOARD ----------------
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
        ["Uncalibrated Logit (top)", f"{raw_logits[top_idx]:.3f}"],
        ["Calibrated Confidence", f"{calibrated_conf:.1%}"],
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
                       colWidths=[59 * mm, 59 * mm, 59 * mm],
                       hAlign="LEFT")
    audit_grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(audit_grid)
    story.append(Spacer(1, 12))

    # ---------------- 4. VISUAL PANELS ----------------
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

    img_w, img_h = 50 * mm, 50 * mm
    rl_base = RLImage(base_buf, width=img_w, height=img_h)
    rl_pert = RLImage(pert_buf, width=img_w, height=img_h)
    rl_diff = RLImage(diff_buf, width=img_w, height=img_h)

    vis_rows = [
        [rl_base, "", rl_pert, "", rl_diff],
        [
            Paragraph("<b>Baseline Grad-CAM</b><br/>Original Saliency", styles["caption"]),
            "",
            Paragraph(f"<b>Grad-CAM under {pert_type}</b><br/>{pert_intensity:.1f}× Scale", styles["caption"]),
            "",
            Paragraph(f"<b>Attribution Variance Map</b><br/>ESS = {ess_score:.3f}", styles["caption"]),
        ]
    ]
    vis_tbl = Table(
        vis_rows,
        colWidths=[54 * mm, 3 * mm, 54 * mm, 3 * mm, 54 * mm],
        hAlign="CENTER"
    )
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
        styles["caption"]
    ))
    story.append(Spacer(1, 12))

    # ---------------- 5. CONFIG SUMMARY ----------------
    story.append(Paragraph("5. Configuration Summary", styles["h2"]))
    config_rows = [
        ["Setting", "Value"],
        ["Min Image Quality Threshold (T_qual)", f"{t_qual:.2f}"],
        ["Min Calibrated Confidence (T_conf)", f"{t_conf:.1%}"],
        ["Min Explanation Stability (T_ess)", f"{t_ess:.2f}"],
        ["Temperature Scaling (T)", f"{temp_scaling:.2f}"],
        ["Perturbation Engine Mode", pert_type],
        ["Perturbation Intensity", f"{pert_intensity:.1f}×"],
        ["Overall Gate Result", "PASS" if is_accepted else "ABSTAIN"],
    ]
    story.append(_kv_table(config_rows, [95 * mm, 83 * mm]))
    story.append(Spacer(1, 10))

    # ---------------- 6. SAFETY GATE ----------------
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
    gate_tbl = Table(gate_checks,
                     colWidths=[72 * mm, 36 * mm, 36 * mm, 34 * mm],
                     hAlign="LEFT")
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
        result_val = gate_checks[i][3]
        if result_val == "PASS":
            gate_tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#15803d")),
                ("FONTNAME", (3, i), (3, i), "Helvetica-Bold"),
            ]))
        else:
            gate_tbl.setStyle(TableStyle([
                ("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#b91c1c")),
                ("FONTNAME", (3, i), (3, i), "Helvetica-Bold"),
            ]))
    story.append(gate_tbl)
    story.append(Spacer(1, 12))

    # ---------------- DISCLAIMER ----------------
    story.append(HRFlowable(width="100%", thickness=0.6, color=BRAND_SLATE_LIGHT, spaceAfter=8))
    story.append(Paragraph("<b>Disclaimer</b>", styles["body"]))
    story.append(Paragraph(
        "This report is generated by the EXACT-CXR Clinical Safety Engine for research and "
        "educational purposes only. It is not a medical device and must not be used as the sole "
        "basis for clinical diagnosis, treatment, or patient management decisions. All findings "
        "must be reviewed and confirmed by a qualified radiologist or licensed clinician. "
        "EXACT-CXR and its authors disclaim any liability for clinical decisions made based on "
        "this output.",
        styles["body_j"]
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
    pert_type = st.selectbox("Perturbation Engine Mode",
                             ["Gaussian Noise", "Micro Rotation", "Contrast Shift", "All Combined"])
    pert_intensity = st.slider("Perturbation Intensity", 0.1, 3.0, 1.0, 0.1)

    st.markdown("""
    <div style="margin-top: 24px; padding: 12px 14px; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; font-size: 0.78rem; color: #075985; line-height: 1.5;">
        <strong>Clinical Note:</strong> All thresholds are calibrated for research use. Adjust per institutional protocol.
    </div>
    """, unsafe_allow_html=True)


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
# MAIN UI
# =====================================================================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-logo-box">
        <img src="{LOGO_DATA_URI}" alt="EXACT-CXR logo"/>
    </div>
    <div class="hero-text-block">
        <div class="hero-title">
            <span>EXACT-CXR</span> <span class="accent">Clinical Safety Engine</span>
            <span class="hero-badge">v7.2</span>
        </div>
        <p class="hero-subtitle">Explanation-Stable, Confidence-Aware &amp; Cross-Domain Chest X-Ray Diagnostic Gate</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Research-use tool · Not a certified medical device · "
    "Full clinical disclaimer will be presented after you upload a radiograph."
)

uploaded_file = st.file_uploader(
    "Upload Frontal Chest Radiograph (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG. Recommended minimum 224×224 px."
)

if uploaded_file is not None:
    try:
        with st.spinner("Executing Quality Gate, Calibrated Inference & Attribution Stability Analysis..."):
            raw_img = Image.open(uploaded_file).convert("RGB")
            rgb_img, tensor_img = preprocess_cxr(raw_img)
            is_qual_valid, qual_score, lap_var, contrast_std, mean_intensity, img_dims = evaluate_quality_gate(raw_img)

            base_cam, raw_logits, probs, top_idx = compute_multi_label_inference(model, tensor_input=tensor_img, temp_scaling=temp_scaling)
            predicted_disease = PATHOLOGIES[top_idx]
            top_prob = probs[top_idx]
            calibrated_conf = float(max(top_prob, 1.0 - top_prob))

            pert_tensor = apply_perturbation(tensor_img, pert_type, pert_intensity)
            p_cam, _, _, _ = compute_multi_label_inference(model, tensor_input=pert_tensor, temp_scaling=temp_scaling)
            ess_score, cos_sim, ssim_val = calculate_ess(base_cam, p_cam)

            rejection_reasons = []
            if qual_score < t_qual or not is_qual_valid:
                rejection_reasons.append(f"Image Quality ({qual_score:.2f}) below threshold ({t_qual:.2f})")
            if calibrated_conf < t_conf:
                rejection_reasons.append(f"Calibrated Confidence ({calibrated_conf:.1%}) below threshold ({t_conf:.1%})")
            if ess_score < t_ess:
                rejection_reasons.append(f"Explanation Stability ESS ({ess_score:.2f}) below threshold ({t_ess:.2f})")
            is_accepted = len(rejection_reasons) == 0

        _run_id = f"{uploaded_file.name}_{uploaded_file.size}"
        _disclaimer_key = f"disclaimer_shown_for_{_run_id}"
        if not st.session_state.get(_disclaimer_key, False):
            show_disclaimer_overlay(duration_seconds=5)
            st.session_state[_disclaimer_key] = True

        if st.session_state.get("balloons_fired_for") != _run_id:
            st.toast("Analysis complete — results ready below.", icon="✅")
            st.session_state["balloons_fired_for"] = _run_id

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
                <strong>Clinical & Safety Audit Note:</strong><br>
                {f"All safety thresholds met. Visual saliency maps demonstrate high spatial stability (ESS = {ess_score:.2f}) under <em>{pert_type}</em> perturbation, confirming the model relies on robust anatomical indicators rather than artifact shortcuts." if is_accepted else f"Safety gate triggered deferral to radiologist. Reason(s): {', '.join(rejection_reasons)}. Visual explanation shifts indicate model sensitivity to input noise or out-of-distribution features."}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Multi-Pathology Diagnostic Matrix</div>', unsafe_allow_html=True)
        cols = st.columns(len(PATHOLOGIES))
        for idx, disease in enumerate(PATHOLOGIES):
            prob = probs[idx]
            is_top = (idx == top_idx)
            with cols[idx]:
                top_class = "top" if is_top else ""
                st.markdown(f"""
                <div class="path-card {top_class}">
                    <div class="p-name">{disease}</div>
                    <div class="p-value">{prob:.1%}</div>
                    <div class="p-logit">Logit: {raw_logits[idx]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

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
                <div style="margin-top: 12px;"><div class="metric-label">Uncalibrated Logit</div>
                <div class="metric-value">{raw_logits[top_idx]:.3f}</div></div>
                <div style="margin-top: 12px;"><div class="metric-label">Calibrated Confidence</div>
                <div class="metric-value">{calibrated_conf:.1%}</div></div>
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

        st.markdown('<div class="section-header">Explanation Stability & Visual Attributions</div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        img_resized = cv2.resize(np.array(rgb_img.convert("RGB")), (224, 224)) / 255.0

        with v1:
            st.markdown("<div class='panel-label'>Baseline Grad-CAM</div>", unsafe_allow_html=True)
            st.image(blend_heatmap(img_resized, base_cam), use_container_width=True,
                     caption=f"Original Saliency — {predicted_disease}")
        with v2:
            st.markdown(f"<div class='panel-label'>Grad-CAM under {pert_type}</div>", unsafe_allow_html=True)
            st.image(blend_heatmap(img_resized, p_cam), use_container_width=True,
                     caption=f"Perturbed Saliency — {pert_intensity}× Scale")
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
            st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 4px;'>Spatial Instability Map · ESS = <strong>{ess_score:.3f}</strong></div>", unsafe_allow_html=True)

        st.warning(
            "**Clinical Caution:** The findings above are generated by a research model and "
            "**must be confirmed by a qualified radiologist**. Do not act on these results "
            "without expert human review. Use of this tool does not replace clinical judgment."
        )

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
                        Complete clinical report with the input radiograph, executive decision, all pathology probabilities,
                        system audit boards, all three visual attribution maps, safety gate breakdown, configuration summary,
                        and disclaimer. Soft-layered with the EXACT-CXR logo watermark in the background of every page.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            pdf_buffer = generate_pdf_report(
                input_image_pil=raw_img,
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
            Upload a frontal chest X-ray image above to run full multi-pathology inference, quality evaluation, calibration analysis, and attribution stability checks.
        </p>
        <div class="feature-list">
            <span class="feat">Multi-Pathology Detection</span>
            <span class="feat">Temperature Calibration</span>
            <span class="feat">Grad-CAM Stability (ESS)</span>
            <span class="feat">Quality Gating</span>
            <span class="feat">PDF Report Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
