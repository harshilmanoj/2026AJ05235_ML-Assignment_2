import json
import io
import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
METADATA_FILE = BASE_DIR / "model_metadata.json"

# The application uses a fixed dark appearance.
DARK_MODE = True


# STYLING
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --canvas: #f6f7f9;
        --paper: #ffffff;
        --ink: #17243a;
        --muted: #738198;
        --line: #e7ebf1;
        --teal: #168f83;
        --teal-soft: #eaf6f3;
        --navy: #142238;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 2%, rgba(190,218,220,.24), transparent 27%),
            radial-gradient(ellipse at 5% 40%, rgba(220,228,239,.22), transparent 30%),
            var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(246,247,249,.82); }
    [data-testid="stToolbar"] { right: 1.5rem; }
    .block-container {
        width: 100%;
        max-width: 100%;
        padding: 1.35rem clamp(1rem, 3.2vw, 3.25rem) 3.5rem;
        box-sizing: border-box;
    }
    h1, h2, h3, h4 {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        letter-spacing: -.035em;
    }
    h1 { font-size: clamp(2rem, 3.5vw, 3rem) !important; font-weight: 800 !important; }
    h2 { font-size: 1.65rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 700 !important; }
    p, label, .stCaption { color: var(--muted); }


    ::selection {
        background: rgba(226,182,108,.38);
        color: #fff8e9;
    }
    html { scroll-behavior: smooth; scroll-padding-top: 110px; }
    .scroll-anchor { scroll-margin-top: 112px; height: 1px; }
    .glass-nav {
        position: sticky;
        top: 8px;
        z-index: 999;
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: space-between;
        gap: .35rem;
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        padding: .30rem;
        margin: .65rem 0 .65rem;
        border: 1px solid rgba(180,190,205,.25);
        border-radius: 19px;
        background: rgba(16,25,40,.74);
        backdrop-filter: blur(24px) saturate(175%);
        -webkit-backdrop-filter: blur(24px) saturate(175%);
        box-shadow: 0 12px 34px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.075);
    }
    .glass-nav .nav-link {
        display: inline-flex;
        flex: 1 1 0;
        min-width: 0;
        align-items: center;
        justify-content: center;
        padding: .62rem .72rem;
        border-radius: 11px;
        color: #aab6c8 !important;
        text-decoration: none !important;
        font-size: clamp(.86rem, .98vw, .96rem);
        font-weight: 700;
        white-space: nowrap;
        text-align: center;
        transition: background .22s ease, color .22s ease, transform .22s ease, box-shadow .22s ease;
    }
    .glass-nav .nav-link:hover {
        color: #147f77 !important; background: rgba(22,143,131,.09);
        transform: translateY(-1px);
    }
    .glass-nav + * {
        margin-top: .25rem !important;
    }
    .glass-nav .nav-link.active {
        color: #fff !important;
        background: linear-gradient(135deg,#1c344c,#28515b);
        box-shadow: 0 4px 12px rgba(20,42,62,.16);
    }
    .feature-link, .workflow-link { display:block; text-decoration:none !important; color:inherit !important; cursor:pointer; }
    .feature-link:hover, .workflow-link:hover { text-decoration:none !important; }
    .workflow-link:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(22,39,63,.08); }
    .feature-arrow { float:right; color:#168f83; font-size:.85rem; }

    .masthead {
        display:flex; align-items:center; justify-content:space-between;
        gap:1rem; padding:.25rem 0 1.15rem;
        border-bottom:1px solid #e3e8ee; margin-bottom:1.1rem;
    }
    .mast-brand {
        display:flex; align-items:center; gap:.75rem;
        color:#18263b; font-size:1.02rem; font-weight:850;
        letter-spacing:.10em;
    }
    .mast-icon {
        display:inline-flex; align-items:center; justify-content:center;
        width:38px; height:38px; border-radius:13px;
        color:white; background:linear-gradient(145deg,#1c8d84,#72c7b5);
        box-shadow:0 5px 14px rgba(22,143,131,.18); font-size:1rem;
    }
    .mast-meta {
        color:#8995a6; font-size:.72rem; font-weight:700;
        letter-spacing:.09em; text-transform:uppercase; text-align:right;
    }
    .student-line {
        display:flex; flex-wrap:wrap; gap:.55rem 1.5rem;
        padding:.85rem 0 1.05rem; color:#78869a; font-size:.79rem;
    }
    .student-line strong { color:#27364c; font-weight:700; }

    /* Native segmented control: clean, horizontal, no radio indicators */
    div[data-testid="stSegmentedControl"] {
        width:100%; margin:.15rem 0 1.65rem;
    }
    div[data-testid="stSegmentedControl"] > label { display:none; }
    div[data-testid="stSegmentedControl"] [role="radiogroup"] {
        display:flex; flex-wrap:wrap; gap:.35rem;
        width:fit-content; max-width:100%;
        padding:.35rem; border:1px solid #e5eaf0;
        border-radius:15px; background:rgba(255,255,255,.82);
        box-shadow:0 5px 20px rgba(24,42,65,.045);
    }
    div[data-testid="stSegmentedControl"] button {
        border:0 !important; border-radius:11px !important;
        padding:.58rem 1rem !important; color:#718096 !important;
        background:transparent !important; font-weight:650 !important;
        transition:all .2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        color:#fff !important; background:#17283f !important;
        box-shadow:0 3px 10px rgba(20,34,56,.14);
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color:#147c75 !important; background:#edf5f4 !important;
    }

    .home-hero {
        position:relative; overflow:hidden;
        display:grid; grid-template-columns:minmax(0,1.35fr) minmax(250px,.65fr);
        gap:2rem; align-items:center;
        padding:2.6rem 2.65rem; margin:.25rem 0 1.35rem;
        border-radius:25px; color:#fff;
        background:
            radial-gradient(circle at 88% 14%,rgba(80,184,164,.22),transparent 28%),
            linear-gradient(118deg,#142238 0%,#1a354b 58%,#214a57 100%);
        box-shadow:0 20px 48px rgba(22,39,61,.13);
    }
    .home-kicker {
        color:#79d7c4; font-size:.7rem; font-weight:800;
        letter-spacing:.16em; text-transform:uppercase; margin-bottom:1rem;
    }
    .home-title {
        color:#fff; font-family:'Manrope',sans-serif;
        font-size:clamp(2.15rem,4.3vw,3.75rem);
        line-height:1.08; letter-spacing:-.055em;
        font-weight:800; margin:0 0 1rem;
    }
    .home-title span { color:#8be0ce; }
    .home-copy {
        color:#c7d3df; font-size:1rem; line-height:1.75;
        max-width:650px; margin:0;
    }
    .hero-side {
        padding:1.25rem; border:1px solid rgba(220,240,242,.16);
        border-radius:19px; background:rgba(255,255,255,.065);
        backdrop-filter:blur(12px);
    }
    .hero-side-label {
        color:#9eb6c6; font-size:.66rem; font-weight:800;
        letter-spacing:.14em; text-transform:uppercase; margin-bottom:1rem;
    }
    .flow-step {
        display:flex; align-items:center; gap:.75rem;
        padding:.7rem 0; border-top:1px solid rgba(220,240,242,.12);
        color:#edf5f8; font-size:.86rem; font-weight:600;
    }
    .flow-number {
        display:inline-flex; align-items:center; justify-content:center;
        width:27px; height:27px; border-radius:9px;
        background:rgba(111,214,194,.15); color:#8be0ce;
        font-size:.72rem; font-weight:800; flex-shrink:0;
    }
    .home-student {
        display:flex; flex-wrap:wrap; align-items:center;
        justify-content:space-between; gap:1rem;
        padding:1rem 1.2rem; margin-bottom:2rem;
        background:rgba(255,255,255,.78); border:1px solid var(--line);
        border-radius:15px;
    }
    .student-item { display:flex; flex-direction:column; gap:.2rem; }
    .student-label {
        color:#8995a6; font-size:.64rem; font-weight:800;
        letter-spacing:.12em; text-transform:uppercase;
    }
    .student-value { color:#26364c; font-size:.88rem; font-weight:700; }
    .section-kicker {
        color:#16877e; font-size:.68rem; font-weight:800;
        letter-spacing:.16em; text-transform:uppercase; margin-bottom:.45rem;
    }
    .section-intro {
        color:#7a879a; font-size:.91rem; line-height:1.65;
        margin-top:-.25rem; margin-bottom:1.25rem;
    }
    .feature-grid {
        display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:.85rem; margin:1rem 0 2rem;
    }
    .feature-card {
        background:rgba(255,255,255,.9); border:1px solid #e6ebf1;
        border-radius:16px; padding:1.15rem; min-height:160px;
        box-shadow:0 7px 22px rgba(22,39,63,.035);
        transition:transform .2s ease, box-shadow .2s ease;
    }
    .feature-card:hover { transform:translateY(-3px); box-shadow:0 12px 28px rgba(22,39,63,.07); }
    .feature-icon {
        display:flex; align-items:center; justify-content:center;
        width:35px; height:35px; border-radius:11px;
        background:#eaf6f3; color:#16877e; font-size:1rem;
        margin-bottom:.85rem;
    }
    .feature-title { color:#26364c; font-size:.91rem; font-weight:750; margin-bottom:.35rem; }
    .feature-copy { color:#7a879a; font-size:.78rem; line-height:1.55; }
    .workflow {
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:.85rem; margin:1rem 0 1.5rem;
    }
    .workflow-card {
        padding:1.05rem 1.15rem; background:#fff;
        border:1px solid #e7ebf1; border-radius:14px;
    }
    .workflow-no { color:#168f83; font-size:.68rem; font-weight:800; letter-spacing:.1em; }
    .workflow-title { color:#26364c; font-weight:750; margin:.35rem 0; }
    .workflow-copy { color:#7a879a; font-size:.8rem; line-height:1.55; }

    .upload-panel {
        background:linear-gradient(110deg,#fff,#fbfcfd);
        border:1px solid #e4e9ef; border-radius:16px;
        padding:1.05rem 1.25rem .8rem;
        box-shadow:0 7px 22px rgba(24,42,65,.035);
        margin:1.1rem 0 .6rem;
    }
    .upload-kicker {
        color:#16877e; font-size:.67rem; font-weight:800;
        letter-spacing:.14em; text-transform:uppercase; margin-bottom:.25rem;
    }
    .upload-title { color:#1c2a40; font-size:1rem; font-weight:750; margin-bottom:.2rem; }
    .upload-copy { color:#8390a2; font-size:.81rem; line-height:1.5; }
    [data-testid="stFileUploader"] section {
        background:#fbfcfd; border:1px dashed #a8c5c2;
        border-radius:13px; padding:.65rem .8rem;
    }
    [data-testid="stFileUploader"] section:hover { border-color:#168f83; background:#f3f9f8; }

    .eyebrow {
        color:#16877e; font-size:.69rem; font-weight:800;
        letter-spacing:.16em; text-transform:uppercase; margin-bottom:.55rem;
    }
    .section-note {
        color:#7a879a; font-size:.92rem; margin-top:-.45rem;
        margin-bottom:1.35rem; line-height:1.6;
    }
    div[data-testid="stMetric"] {
        background:rgba(255,255,255,.94); border:1px solid #e6eaf0;
        border-radius:14px; padding:1rem 1.1rem;
        box-shadow:0 6px 18px rgba(22,39,63,.035);
    }
    div[data-testid="stMetricLabel"] { color:#718096; font-size:.82rem; }
    div[data-testid="stMetricValue"] { color:#1b2a40; font-weight:700; }
    div[data-testid="stDataFrame"] { border:1px solid #e4e8ef; border-radius:12px; overflow:hidden; }
    .stAlert { border-radius:12px; }
    .stButton button, .stDownloadButton button {
        border-radius:10px; border:1px solid #dce5e9;
        background:#17283f; color:#fff; font-weight:650;
        padding:.55rem 1rem; transition:all .18s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color:#168f83; background:#205b5b; color:#fff;
    }

    /* Refined model selector — soft glass surface, not a full-width dark slab */
    div[data-testid="stSelectbox"] {
        max-width: 100%;
    }
    div[data-testid="stSelectbox"] > label {
        color: #66758b !important;
        font-size: .76rem !important;
        font-weight: 750 !important;
        letter-spacing: .10em !important;
        text-transform: uppercase !important;
        margin-bottom: .48rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 54px !important;
        border-radius: 15px !important;
        border: 1px solid rgba(151, 166, 184, .30) !important;
        background: rgba(255,255,255,.72) !important;
        box-shadow:
            0 8px 24px rgba(24,42,65,.055),
            inset 0 1px 0 rgba(255,255,255,.88) !important;
        backdrop-filter: blur(18px) saturate(150%);
        -webkit-backdrop-filter: blur(18px) saturate(150%);
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        border-color: rgba(22,143,131,.55) !important;
        box-shadow: 0 10px 28px rgba(22,143,131,.09), inset 0 1px 0 rgba(255,255,255,.9) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #168f83 !important;
        box-shadow: 0 0 0 3px rgba(22,143,131,.12), 0 8px 24px rgba(24,42,65,.06) !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    div[data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #24344a !important;
        font-weight: 650 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: #718096 !important;
        fill: #718096 !important;
    }
    div[data-baseweb="popover"] ul[role="listbox"] {
        border: 1px solid rgba(151,166,184,.28) !important;
        border-radius: 14px !important;
        padding: .35rem !important;
        background: rgba(255,255,255,.96) !important;
        box-shadow: 0 18px 48px rgba(17,31,51,.16) !important;
        backdrop-filter: blur(22px);
    }
    div[data-baseweb="popover"] li[role="option"] {
        border-radius: 9px !important;
        color: #26364c !important;
        font-weight: 550 !important;
        transition: background .15s ease;
    }
    div[data-baseweb="popover"] li[role="option"]:hover {
        background: #edf6f4 !important;
        color: #147f77 !important;
    }

    .model-picker-heading {
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        letter-spacing: -.025em;
        margin-bottom: .25rem;
    }
    .model-picker-copy {
        max-width: 720px;
        line-height: 1.55 !important;
        margin-bottom: .35rem;
    }
    div[data-testid="stSelectbox"] {
        padding: 0 0 .20rem;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] {
        gap: .12rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        cursor: pointer;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        transition: all .22s ease !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        transform: translateY(-1px);
    }

    .model-picker-heading {
        color: #24344a;
        font-size: 1.02rem;
        font-weight: 750;
        letter-spacing: -.02em;
        margin-bottom: .18rem;
    }
    .model-picker-copy {
        color: #8290a3;
        font-size: .82rem;
        line-height: 1.5;
    }
    .model-picker-meta {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: .85rem 1rem;
        border: 1px solid rgba(151,166,184,.22);
        border-radius: 14px;
        background: rgba(255,255,255,.48);
    }
    .model-picker-meta-label {
        color: #8995a6;
        font-size: .62rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .model-picker-meta-value {
        color: #26364c;
        font-size: .92rem;
        font-weight: 750;
        margin-top: .2rem;
    }


    /* Dedicated visual treatment around the classifier selection area */
    .st-key-model_picker_panel {
        border-radius: 22px !important;
        padding: 1.35rem 1.45rem 1.05rem !important;
        margin: .8rem 0 1.35rem !important;
        border: 1px solid rgba(226,182,108,.42) !important;
        background:
            radial-gradient(circle at 4% 0%, rgba(226,182,108,.10), transparent 38%),
            linear-gradient(135deg, rgba(255,255,255,.075), rgba(255,255,255,.018)) !important;
        box-shadow:
            0 16px 42px rgba(0,0,0,.17),
            inset 0 1px 0 rgba(255,255,255,.07) !important;
        backdrop-filter: blur(20px) saturate(145%);
        -webkit-backdrop-filter: blur(20px) saturate(145%);
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    .st-key-model_picker_panel:hover {
        border-color: rgba(226,182,108,.68) !important;
        box-shadow:
            0 18px 48px rgba(0,0,0,.20),
            0 0 0 1px rgba(226,182,108,.08),
            inset 0 1px 0 rgba(255,255,255,.08) !important;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] > label {
        color: #e8bd73 !important;
        font-size: .76rem !important;
        font-weight: 850 !important;
        letter-spacing: .14em !important;
        margin: .18rem 0 .28rem !important;
        line-height: 1.25 !important;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 68px !important;
        border: 1.5px solid rgba(240,204,139,.78) !important;
        border-radius: 17px !important;
        background: linear-gradient(115deg, rgba(39,53,74,.98), rgba(24,37,57,.98)) !important;
        box-shadow:
            0 10px 28px rgba(0,0,0,.24),
            0 0 0 3px rgba(226,182,108,.10),
            inset 0 1px 0 rgba(255,255,255,.12) !important;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        border-color: #f3d59e !important;
        box-shadow:
            0 12px 32px rgba(0,0,0,.27),
            0 0 0 4px rgba(226,182,108,.14),
            inset 0 1px 0 rgba(255,255,255,.14) !important;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #F0CC8B !important;
        font-size: 1.12rem !important;
        font-weight: 800 !important;
    }

    /* Selected classifier: warm champagne treatment to match the dashboard */
    .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: linear-gradient(
            110deg,
            rgba(75, 61, 43, .72),
            rgba(38, 43, 54, .98) 58%,
            rgba(30, 40, 55, .98)
        ) !important;
        border-color: rgba(232, 189, 115, .92) !important;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, .24),
            0 0 0 3px rgba(232, 189, 115, .13),
            inset 0 1px 0 rgba(255, 239, 207, .14) !important;
    }
    .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: #F0CC8B !important;
        fill: #F0CC8B !important;
    }
    div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
        background: rgba(232, 189, 115, .20) !important;
        color: #F0CC8B !important;
    }
    .st-key-model_picker_panel .model-picker-heading {
        font-size: 1.14rem !important;
        font-weight: 850 !important;
        letter-spacing: -.035em !important;
    }
    .st-key-model_picker_panel .model-picker-copy {
        font-size: .88rem !important;
        max-width: 650px;
    }
    .st-key-model_picker_panel .model-picker-meta {
        min-height: 105px;
        border-color: rgba(226,182,108,.24) !important;
        background: rgba(11,19,32,.42) !important;
        border-radius: 16px !important;
    }
    .st-key-model_picker_panel .model-picker-meta-value {
        color: #f0cc8b !important;
        font-size: 1rem !important;
    }

    @media (max-width: 760px) {
        .st-key-model_picker_panel { padding: 1rem !important; }
    }


    @media (max-width: 900px) {
        .glass-nav {
            flex-wrap: wrap;
            justify-content: stretch;
            gap: .3rem;
        }
        .glass-nav .nav-link {
            flex: 1 1 calc(33.333% - .3rem);
            white-space: normal;
            min-height: 44px;
            padding: .55rem .42rem;
            font-size: .84rem;
        }
    }
    @media (max-width: 560px) {
        .block-container {
            padding-left: .85rem !important;
            padding-right: .85rem !important;
        }
        .glass-nav {
            position: sticky;
            top: 4px;
            padding: .28rem;
            border-radius: 15px;
            margin-bottom: .5rem;
        }
        .glass-nav .nav-link {
            flex: 1 1 calc(50% - .3rem);
            font-size: .80rem;
            min-height: 40px;
            padding: .5rem .35rem;
        }
        .masthead { align-items: flex-start; }
        .student-line { gap: .5rem .9rem; }
        .st-key-model_picker_panel { padding: .85rem !important; }
    }


    /* Compact classification report, aligned with the paired PR chart */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(226,182,108,.30) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Option 5: class-wise performance cards with compact metric bars */
    .class-report-heading { margin: .05rem 0 .8rem; }
    .class-report-kicker {
        color: #B48A4B; font-size: .65rem; font-weight: 850;
        letter-spacing: .15em; margin-bottom: .22rem;
    }
    .class-report-title {
        color: #24344A; font-family: 'Manrope', sans-serif;
        font-size: 1.18rem; line-height: 1.2; font-weight: 850;
        letter-spacing: -.035em;
    }
    .class-report-subtitle {
        color: #8290A3; font-size: .78rem; margin-top: .25rem; line-height: 1.4;
    }
    .class-performance-grid {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .55rem; align-items: stretch;
    }
    .class-performance-card {
        min-width: 0; border: 1px solid rgba(226,182,108,.34);
        border-radius: 14px; padding: .78rem .72rem .72rem;
        background: radial-gradient(circle at 0% 0%, rgba(232,189,115,.12), transparent 48%),
                    linear-gradient(145deg, #17263B, #101C30);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.055), 0 8px 22px rgba(0,0,0,.12);
    }
    .class-card-topline {
        color: #B9A27D; font-size: .58rem; font-weight: 850; letter-spacing: .13em;
    }
    .class-card-name {
        color: #F0F3F8; font-size: .93rem; font-weight: 800;
        margin-top: .25rem; line-height: 1.2; overflow-wrap: anywhere;
    }
    .class-f1-value {
        color: #F0CC8B; font-size: 1.48rem; font-weight: 900;
        letter-spacing: -.045em; line-height: 1.1; margin-top: .5rem; white-space: nowrap;
    }
    .class-f1-label {
        color: #AAB6C8; font-size: .57rem; font-weight: 850;
        letter-spacing: .13em; margin-top: .12rem;
    }
    .class-card-divider { height: 1px; background: rgba(190,204,222,.14); margin: .58rem 0; }
    .class-metric { margin: .5rem 0; }
    .class-metric-line {
        display: flex; justify-content: space-between; align-items: baseline;
        gap: .25rem; color: #B9C5D5; font-size: .65rem; line-height: 1.25;
    }
    .class-metric-line strong { color: #F2D49C; font-size: .67rem; white-space: nowrap; }
    .class-progress-track {
        height: 5px; background: #34445A; border-radius: 99px;
        overflow: hidden; margin-top: .25rem;
    }
    .class-progress-fill {
        height: 100%; border-radius: 99px;
        background: linear-gradient(90deg, #C99A55, #F0CC8B);
    }
    .class-support {
        display: flex; align-items: baseline; flex-wrap: wrap; gap: .28rem;
        border-top: 1px solid rgba(190,204,222,.13);
        margin-top: .58rem; padding-top: .5rem;
    }
    .class-support span {
        width: 100%; color: #92A2B8; font-size: .55rem;
        font-weight: 850; letter-spacing: .12em;
    }
    .class-support strong { color: #F0CC8B; font-size: .94rem; font-weight: 850; }
    .class-support small { color: #9AA9BE; font-size: .59rem; }
    .class-overall-card {
        display: grid; grid-template-columns: 1fr auto; gap: .2rem .5rem;
        align-items: center; border: 1px solid rgba(226,182,108,.48);
        border-radius: 13px; padding: .65rem .78rem .62rem; margin-top: .62rem;
        background: linear-gradient(115deg, rgba(232,189,115,.16), rgba(21,34,53,.96));
    }
    .class-overall-left, .class-overall-right { display: flex; flex-direction: column; gap: .05rem; }
    .class-overall-label, .class-overall-right span {
        color: #B9A27D; font-size: .56rem; font-weight: 850; letter-spacing: .11em;
    }
    .class-overall-left strong { color: #F0CC8B; font-size: 1.38rem; line-height: 1.1; font-weight: 900; }
    .class-overall-right { text-align: right; }
    .class-overall-right strong { color: #F2F4F8; font-size: .98rem; }
    .class-overall-track {
        grid-column: 1 / -1; height: 5px; border-radius: 99px;
        background: #34445A; overflow: hidden; margin-top: .35rem;
    }
    .class-overall-track div {
        height: 100%; background: linear-gradient(90deg, #C99A55, #F0CC8B); border-radius: 99px;
    }
    @media (max-width: 1050px) {
        .class-performance-grid { gap: .4rem; }
        .class-performance-card { padding: .65rem .55rem; }
        .class-f1-value { font-size: 1.25rem; }
    }
    @media (max-width: 720px) {
        .class-performance-grid { grid-template-columns: 1fr; }
        .class-performance-card { padding: .85rem; }
        .class-f1-value { font-size: 1.45rem; }
    }


    .precision-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0 .55rem;
        font-family: 'DM Sans', sans-serif;
    }
    .precision-stat-card {
        min-width: 0;
        padding: 1.05rem 1.2rem 1rem;
        border: 1px solid var(--line);
        border-left: 3px solid #D6A95D;
        border-radius: 16px;
        background: var(--paper);
        box-shadow: 0 7px 20px rgba(15, 30, 50, .055);
    }
    .precision-stat-card.phishing { border-left-color: #83A99A; }
    .precision-stat-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .65rem;
        margin-bottom: .45rem;
    }
    .precision-stat-class {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: .94rem;
        line-height: 1.35;
        font-weight: 700;
        letter-spacing: -.015em;
    }
    .precision-stat-tag {
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: .61rem;
        font-weight: 700;
        letter-spacing: .075em;
        white-space: nowrap;
    }
    .precision-stat-value {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 2.25vw, 2.05rem);
        line-height: 1.22;
        font-weight: 750;
        letter-spacing: -.035em;
        font-variant-numeric: tabular-nums;
    }
    .precision-stat-detail {
        margin-top: .4rem;
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: .79rem;
        line-height: 1.55;
    }
    .precision-stat-detail strong {
        color: var(--ink);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .precision-stat-track {
        height: 5px;
        margin-top: .75rem;
        overflow: hidden;
        border-radius: 99px;
        background: var(--line);
    }
    .precision-stat-track span {
        display: block;
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #C99A55, #F0CC8B);
    }
    .precision-stat-card.phishing .precision-stat-track span {
        background: linear-gradient(90deg, #638D7D, #A6C8B8);
    }

    .validation-check-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .58rem .72rem;
        margin-top: .45rem;
        width: 100%;
    }
    .validation-check-item {
        display: flex;
        align-items: flex-start;
        gap: .62rem;
        min-width: 0;
        padding: .62rem .72rem;
        border: 1px solid var(--line);
        border-radius: 11px;
        background: var(--paper);
    }
    .validation-check-icon {
        display: grid;
        place-items: center;
        flex: 0 0 21px;
        width: 21px;
        height: 21px;
        margin-top: .02rem;
        border-radius: 50%;
        background: rgba(57, 167, 126, .16);
        color: #42B889;
        font-family: 'DM Sans', sans-serif;
        font-size: .82rem;
        font-weight: 900;
        line-height: 1;
    }
    .validation-check-copy {
        display: flex;
        flex-direction: column;
        gap: .08rem;
        min-width: 0;
    }
    .validation-check-name {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: .78rem;
        line-height: 1.4;
        font-weight: 750;
    }
    .validation-check-detail {
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: .72rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }
    @media (max-width: 720px) {
        .validation-check-grid { grid-template-columns: 1fr; }
    }

    .classification-table-scroll {
        width: 100%;
        overflow-x: auto;
        border-radius: 11px;
    }
    .classification-report-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: auto;
        font-size: .74rem;
    }
    .classification-report-table th,
    .classification-report-table td {
        padding: .62rem .52rem;
        border-right: 1px solid #35445A;
        border-bottom: 1px solid #35445A;
        white-space: nowrap;
    }
    .classification-report-table thead th {
        background: #25364D;
        color: #F0CC8B;
        font-size: .65rem;
        font-weight: 850;
        text-align: right;
        letter-spacing: .02em;
    }
    .classification-report-table thead th:first-child,
    .classification-report-table tbody th {
        text-align: left;
    }
    .classification-report-table tbody th {
        color: #F2D49C;
        font-weight: 750;
        background: #142238;
    }
    .classification-report-table tbody td {
        color: #EAF0F7;
        text-align: right;
        background: #111C2D;
        font-variant-numeric: tabular-nums;
    }
    .classification-report-table tbody tr.average-row th,
    .classification-report-table tbody tr.average-row td {
        background: #1B2A40;
        font-weight: 750;
    }
    .classification-report-table tr:last-child th,
    .classification-report-table tr:last-child td {
        border-bottom: 0;
    }
    .classification-report-table th:last-child,
    .classification-report-table td:last-child {
        border-right: 0;
    }
    [data-testid="stExpander"] {
        border-color: rgba(226,182,108,.28) !important;
        border-radius: 11px !important;
    }

    .footer {
        color:#98a2b1; font-size:.73rem; letter-spacing:.04em;
        text-align:center; padding-top:1.3rem;
        border-top:1px solid #e1e6ed; margin-top:3rem;
    }

    [data-testid="stToggle"] label { font-weight:650; color:var(--ink) !important; }
    [data-testid="stToggle"] [role="switch"] {
        border-radius:999px;
    }
    .masthead img { object-fit:contain; }

    @media (max-width: 900px) {
        .home-hero { grid-template-columns:1fr; padding:2rem; }
        .hero-side { max-width:100%; }
        .feature-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width: 600px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .home-hero { padding:1.5rem; border-radius:19px; }
        [data-testid="stImage"] img { max-height:58px; object-fit:contain; }
        .mast-meta { display:none; }
        .feature-grid, .workflow { grid-template-columns:1fr; }
        .glass-nav { width:100%; overflow-x:auto; flex-wrap:nowrap; scrollbar-width:none; }
        .glass-nav::-webkit-scrollbar { display:none; }
        .glass-nav .nav-link { padding:.58rem .78rem; font-size:.79rem; }
        .home-student { align-items:flex-start; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# NIGHT THEME OVERRIDES

if DARK_MODE:
    st.markdown(
        """
        <style>
        :root {
            --canvas: #090e18;
            --paper: #111a29;
            --ink: #edf2fa;
            --muted: #a0aec0;
            --line: #263448;
            --teal: #e2b66c;
            --teal-soft: #2b2a27;
            --navy: #101c30;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 86% 3%, rgba(65,91,130,.18), transparent 30%),
                radial-gradient(ellipse at 7% 35%, rgba(35,69,87,.14), transparent 32%),
                var(--canvas) !important;
            color: var(--ink) !important;
        }
        [data-testid="stHeader"] { background: rgba(9,14,24,.88) !important; }
        [data-testid="stToolbar"] { color: #dce5f2 !important; }
        h1,h2,h3,h4,h5,h6 { color: #edf2fa !important; }
        p, label, .stCaption, [data-testid="stMarkdownContainer"] p {
            color: #aab6c8;
        }
        .masthead { border-bottom-color: #263448 !important; }
        .glass-nav {
            display: flex !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            justify-content: space-between !important;
            background: rgba(16,25,40,.78) !important;
            border-color: rgba(139,157,184,.22) !important;
            box-shadow: 0 12px 36px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.075) !important;
        }
        .glass-nav .nav-link { color:#aab6c8 !important; }
        .glass-nav .nav-link:hover { color:#f0cc8b !important; background:rgba(226,182,108,.10) !important; }
        .glass-nav .nav-link.active {
            color:#182033 !important;
            background:linear-gradient(135deg,#f0cc8b,#d5a45d) !important;
            box-shadow:0 4px 14px rgba(226,182,108,.16) !important;
        }

        /* Classifier picker: same smoked-glass language as the full-width nav */
        .st-key-model_picker_panel {
            background:
                radial-gradient(ellipse at 8% 0%, rgba(240,204,139,.22), transparent 42%),
                radial-gradient(ellipse at 100% 100%, rgba(226,182,108,.09), transparent 38%),
                linear-gradient(125deg, rgba(35,43,57,.99), rgba(17,29,46,.99) 54%, rgba(13,23,38,.99)) !important;
            border: 1.5px solid rgba(232,189,115,.72) !important;
            border-radius: 21px !important;
            box-shadow:
                0 18px 46px rgba(0,0,0,.30),
                0 0 0 1px rgba(240,204,139,.13),
                0 0 26px rgba(226,182,108,.16),
                inset 0 1px 0 rgba(255,239,207,.19) !important;
            backdrop-filter: blur(24px) saturate(155%);
            -webkit-backdrop-filter: blur(24px) saturate(155%);
            transition: border-color .22s ease, box-shadow .22s ease, background .22s ease;
        }

        /* Streamlit renders a separate border wrapper inside keyed containers. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-model_picker_panel) {
            background:
                radial-gradient(ellipse at 8% 0%, rgba(240,204,139,.22), transparent 42%),
                radial-gradient(ellipse at 100% 100%, rgba(226,182,108,.09), transparent 38%),
                linear-gradient(125deg, rgba(35,43,57,.99), rgba(17,29,46,.99) 54%, rgba(13,23,38,.99)) !important;
            border: 1.5px solid rgba(232,189,115,.72) !important;
            border-radius: 21px !important;
            box-shadow:
                0 18px 46px rgba(0,0,0,.30),
                0 0 0 1px rgba(240,204,139,.13),
                0 0 26px rgba(226,182,108,.16),
                inset 0 1px 0 rgba(255,239,207,.19) !important;
            overflow: hidden;
        }
        .st-key-model_picker_panel,
        .st-key-model_picker_panel > div,
        .st-key-model_picker_panel [data-testid="stVerticalBlock"] {
            background: transparent !important;
        }
        .st-key-model_picker_panel:hover {
            border-color: rgba(246,211,151,.92) !important;
            box-shadow:
                0 20px 50px rgba(0,0,0,.32),
                0 0 0 1px rgba(240,204,139,.20),
                0 0 34px rgba(226,182,108,.22),
                inset 0 1px 0 rgba(255,239,207,.22) !important;
        }
        .st-key-model_picker_panel .model-picker-heading {
            display: inline-block;
            width: fit-content;
            color: #f2cc87 !important;
            font-size: clamp(1.32rem, 1.65vw, 1.58rem) !important;
            font-weight: 900 !important;
            letter-spacing: -.045em !important;
            line-height: 1.18 !important;
            margin-bottom: .42rem !important;
            text-shadow: 0 2px 18px rgba(226,182,108,.20);
            background: linear-gradient(100deg, #fff0cf 0%, #f2cc87 48%, #dba95e 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .st-key-model_picker_panel .model-picker-copy { color: #aab7ca !important; }
        .st-key-model_picker_panel .model-picker-meta {
            background: linear-gradient(145deg, rgba(226,182,108,.105), rgba(9,16,28,.48)) !important;
            border: 1px solid rgba(226,182,108,.30) !important;
            box-shadow: inset 0 1px 0 rgba(255,239,207,.07) !important;
        }
        .st-key-model_picker_panel .model-picker-meta-label { color: #9aa9be !important; }
        .st-key-model_picker_panel .model-picker-meta-value { color: #f0cc8b !important; }
        .st-key-model_picker_panel div[data-testid="stSelectbox"] > label {
            color: #e8bd73 !important;
            margin-bottom: .28rem !important;
        }
        .st-key-model_picker_panel div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: linear-gradient(115deg, #263750, #19283e) !important;
            border: 1.5px solid rgba(226,182,108,.62) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,.24), 0 0 0 3px rgba(226,182,108,.075), inset 0 1px 0 rgba(255,255,255,.10) !important;
        }
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] input {
            color: #fffaf0 !important;
        }

        /* Night appearance: smoked-glass model picker */
        div[data-testid="stSelectbox"] > label {
            color: #e0bd82 !important;
            font-size: .74rem !important;
            font-weight: 850 !important;
            letter-spacing: .14em !important;
            margin-bottom: .55rem !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            min-height: 62px !important;
            background: linear-gradient(135deg, rgba(31,46,67,.98), rgba(20,32,50,.98)) !important;
            border: 1px solid rgba(226,182,108,.48) !important;
            border-radius: 16px !important;
            box-shadow:
                0 12px 30px rgba(0,0,0,.25),
                0 0 0 1px rgba(226,182,108,.07),
                inset 0 1px 0 rgba(255,255,255,.09) !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
            border-color: rgba(226,182,108,.58) !important;
            box-shadow: 0 10px 28px rgba(226,182,108,.08), inset 0 1px 0 rgba(255,255,255,.07) !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
            border-color: #dcb36f !important;
            box-shadow: 0 0 0 3px rgba(226,182,108,.13), 0 10px 28px rgba(0,0,0,.2) !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        div[data-testid="stSelectbox"] [data-baseweb="select"] input {
            color: #ffffff !important;
            font-size: 1.04rem !important;
            font-weight: 750 !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: #dcb36f !important;
            fill: #dcb36f !important;
        }
        div[data-baseweb="popover"] ul[role="listbox"] {
            background: rgba(19,29,45,.97) !important;
            border-color: #34445a !important;
            box-shadow: 0 20px 52px rgba(0,0,0,.42) !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            color: #dce5f1 !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover {
            background: rgba(226,182,108,.13) !important;
            color: #f0cc8b !important;
        }
        .model-picker-heading { color: #edf2fa !important; }
        .model-picker-copy { color: #9ba9bd !important; }
        .model-picker-meta {
            background: rgba(17,26,41,.60) !important;
            border-color: #29384d !important;
        }
        .model-picker-meta-label { color: #91a0b5 !important; }
        .model-picker-meta-value { color: #f0cc8b !important; }

        /* Metric selector: champagne tags instead of default bright-red chips */
        div[data-testid="stMultiSelect"] > label {
            color: #AAB6C8 !important;
            font-weight: 650 !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background: rgba(17,28,45,.94) !important;
            border: 1px solid rgba(226,182,108,.34) !important;
            border-radius: 14px !important;
            min-height: 54px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.045) !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background: linear-gradient(135deg, #E8BD73, #D5A45D) !important;
            border: 1px solid rgba(255,231,187,.48) !important;
            border-radius: 9px !important;
            color: #172238 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,.14) !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
            color: #172238 !important;
            font-weight: 750 !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
            fill: #5D4728 !important;
            color: #5D4728 !important;
        }
        div[data-testid="stMultiSelect"] input {
            color: #EDF2FA !important;
        }
        div[data-testid="stMultiSelect"] [data-baseweb="select"] svg {
            color: #E8BD73 !important;
            fill: #E8BD73 !important;
        }
        div[data-baseweb="popover"] ul[role="listbox"] {
            background: #111C2D !important;
            border: 1px solid #35445A !important;
            border-radius: 13px !important;
            box-shadow: 0 18px 44px rgba(0,0,0,.38) !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            color: #DCE5F1 !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover,
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
            background: rgba(232,189,115,.14) !important;
            color: #F0CC8B !important;
        }

        .feature-link { color:inherit !important; }
        .feature-arrow { color:#e2b66c !important; }
        .mast-brand { color: #f3f5f9 !important; }
        .mast-meta, .student-line { color: #8f9db2 !important; }
        .student-line strong { color: #e7edf7 !important; }
        div[data-testid="stSegmentedControl"] [role="radiogroup"] {
            background: rgba(20,30,47,.92) !important;
            border-color: #2b3a50 !important;
            box-shadow: 0 8px 28px rgba(0,0,0,.22) !important;
        }
        div[data-testid="stSegmentedControl"] button {
            color: #aab6c8 !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            color: #172033 !important;
            background: linear-gradient(135deg,#f0cc8b,#d9a95e) !important;
            box-shadow: 0 3px 12px rgba(226,182,108,.18) !important;
        }
        div[data-testid="stSegmentedControl"] button:hover {
            color: #f0cc8b !important;
            background: rgba(226,182,108,.10) !important;
        }
        .home-student, .feature-card, .workflow-card, .upload-panel,
        .status-card {
            background: rgba(17,26,41,.92) !important;
            border-color: #29384d !important;
            box-shadow: 0 10px 28px rgba(0,0,0,.18) !important;
        }
        .student-label, .section-intro, .feature-copy, .workflow-copy,
        .upload-copy, .section-note { color: #9ba9bd !important; }
        .student-value, .feature-title, .workflow-title,
        .upload-title { color: #edf2fa !important; }
        .section-kicker, .eyebrow, .upload-kicker, .workflow-no {
            color: #e2b66c !important;
        }
        .feature-icon {
            background: rgba(226,182,108,.13) !important;
            color: #f0cc8b !important;
        }
        [data-testid="stFileUploader"] section {
            background: #101a2a !important;
            border-color: #53647a !important;
        }
        [data-testid="stFileUploader"] section:hover {
            background: #172437 !important;
            border-color: #d8ad6b !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span { color: #d5deea !important; }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg,#141f31,#101827) !important;
            border-color: #2a3a50 !important;
            box-shadow: 0 8px 24px rgba(0,0,0,.18) !important;
        }
        div[data-testid="stMetricLabel"] { color: #9eacc0 !important; }
        div[data-testid="stMetricValue"] { color: #f2f5fa !important; }
        div[data-testid="stDataFrame"] {
            border-color: #2a3a50 !important;
        }
        [data-testid="stDataFrame"] * { color-scheme: dark; }
        .stAlert {
            background: #172438 !important;
            border-color: #34465f !important;
            color: #e4ebf5 !important;
        }
        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg,#eac27e,#d5a45d) !important;
            color: #172033 !important;
            border-color: #d5a45d !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background: #f3d59e !important;
            color: #111a29 !important;
        }

        /* Download actions: champagne-gold button with dark, readable content */
        .stDownloadButton button,
        .stDownloadButton button:hover,
        .stDownloadButton button:focus,
        .stDownloadButton button:active {
            background: linear-gradient(135deg, #F0CC8B 0%, #D9A95E 100%) !important;
            border: 1px solid rgba(255,231,187,.72) !important;
            color: #172033 !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            box-shadow: 0 7px 20px rgba(226,182,108,.18),
                        inset 0 1px 0 rgba(255,255,255,.30) !important;
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
        }
        .stDownloadButton button p,
        .stDownloadButton button span,
        .stDownloadButton button div,
        .stDownloadButton button svg {
            color: #172033 !important;
            fill: #172033 !important;
            font-weight: 800 !important;
        }
        .stDownloadButton button:hover {
            transform: translateY(-1px);
            filter: brightness(1.06);
            box-shadow: 0 10px 25px rgba(226,182,108,.25),
                        inset 0 1px 0 rgba(255,255,255,.36) !important;
        }

        /* CSV upload: make the actual Browse/Upload control match the gold actions */
        [data-testid="stFileUploader"] section {
            background: linear-gradient(135deg, rgba(226,182,108,.075), rgba(16,26,42,.96)) !important;
            border: 1px dashed rgba(226,182,108,.58) !important;
            border-radius: 15px !important;
        }
        [data-testid="stFileUploader"] section:hover {
            background: linear-gradient(135deg, rgba(226,182,108,.13), rgba(22,35,54,.98)) !important;
            border-color: rgba(240,204,139,.88) !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] button:focus {
            background: linear-gradient(135deg, #F0CC8B 0%, #D9A95E 100%) !important;
            border: 1px solid rgba(255,231,187,.72) !important;
            border-radius: 10px !important;
            color: #172033 !important;
            font-weight: 800 !important;
            box-shadow: 0 5px 16px rgba(226,182,108,.16) !important;
        }
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] button svg {
            color: #172033 !important;
            fill: #172033 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {
            color: #E4D4B7 !important;
        }

        /* FINAL CONTRAST OVERRIDES — keep the selected model visibly gold */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: linear-gradient(120deg, #F0CC8B 0%, #E5B96F 55%, #D9A45B 100%) !important;
            border: 1.5px solid rgba(255, 231, 187, .92) !important;
            border-radius: 14px !important;
            box-shadow:
                0 8px 22px rgba(0, 0, 0, .24),
                0 0 0 3px rgba(232, 189, 115, .13),
                inset 0 1px 0 rgba(255, 255, 255, .38) !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
            background: linear-gradient(120deg, #F6D99F 0%, #EBC47E 55%, #E0B16B 100%) !important;
            border-color: #FFF0CF !important;
            box-shadow:
                0 10px 26px rgba(0, 0, 0, .25),
                0 0 0 4px rgba(232, 189, 115, .18),
                inset 0 1px 0 rgba(255, 255, 255, .48) !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] *,
        div[data-testid="stSelectbox"] [data-baseweb="select"] input,
        div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            font-weight: 850 !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: #172033 !important;
            fill: #172033 !important;
            opacity: 1 !important;
        }
        /* Keep the open options menu dark, with a clear gold selected/hover state. */
        div[data-baseweb="popover"] ul[role="listbox"] {
            background: #111C2D !important;
            border: 1px solid rgba(232, 189, 115, .48) !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            color: #E7D5B5 !important;
        }
        div[data-baseweb="popover"] li[role="option"] *,
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
            color: #F0CC8B !important;
            -webkit-text-fill-color: #F0CC8B !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover,
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
            background: rgba(232, 189, 115, .17) !important;
        }

        /* Upload icon: use an explicit dark stroke and full opacity on gold. */
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #F0CC8B 0%, #D9A95E 100%) !important;
            color: #172033 !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button svg,
        [data-testid="stFileUploader"] button svg *,
        [data-testid="stFileUploader"] button [data-testid="stIconMaterial"] {
            color: #172033 !important;
            fill: none !important;
            stroke: #172033 !important;
            stroke-width: 2.2px !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button svg path,
        [data-testid="stFileUploader"] button svg line,
        [data-testid="stFileUploader"] button svg polyline {
            stroke: #172033 !important;
            stroke-width: 2.2px !important;
            opacity: 1 !important;
        }

        /* Selected classifier value and every item in its dropdown use champagne gold */
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] *,
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
            color: #F0CC8B !important;
            -webkit-text-fill-color: #F0CC8B !important;
            font-weight: 800 !important;
        }
        .st-key-model_picker_panel div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: #F0CC8B !important;
            fill: #F0CC8B !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            color: #E7D5B5 !important;
        }
        div[data-baseweb="popover"] li[role="option"] *,
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
            color: #F0CC8B !important;
            -webkit-text-fill-color: #F0CC8B !important;
        }
        div[data-baseweb="popover"] li[role="option"]:hover {
            background: rgba(232,189,115,.17) !important;
        }
        .footer { color: #78879d !important; border-top-color: #263448 !important; }
        /* Keep the hero rich and cinematic in both appearance modes. */
        .home-hero {
            background:
                radial-gradient(circle at 88% 14%,rgba(202,164,103,.18),transparent 28%),
                linear-gradient(118deg,#101b2d 0%,#172a43 58%,#243b4a 100%) !important;
            box-shadow: 0 24px 65px rgba(0,0,0,.32) !important;
        }
        .hero-side { background: rgba(255,255,255,.045) !important; }

        .classification-report-table thead th {
            background: #25364D !important;
            color: #F0CC8B !important;
        }
        .classification-report-table tbody th {
            background: #142238 !important;
            color: #F2D49C !important;
        }
        .classification-report-table tbody td {
            background: #111C2D !important;
            color: #EAF0F7 !important;
        }
        .classification-report-table tbody tr.average-row th,
        .classification-report-table tbody tr.average-row td {
            background: #1B2A40 !important;
        }
        [data-testid="stExpander"] {
            background: rgba(17,28,45,.54) !important;
            border-color: rgba(226,182,108,.30) !important;
        }


        .class-report-kicker { color: #E8BD73 !important; }
        .class-report-title { color: #F0CC8B !important; }
        .class-report-subtitle { color: #AAB6C8 !important; }
        .class-performance-card {
            border-color: rgba(226,182,108,.36) !important;
            background: radial-gradient(circle at 0% 0%, rgba(232,189,115,.13), transparent 48%),
                        linear-gradient(145deg, #17263B, #101C30) !important;
        }
        .class-card-name { color: #F0F3F8 !important; }
        .class-f1-value, .class-metric-line strong, .class-support strong { color: #F0CC8B !important; }
        .class-metric-line { color: #B9C5D5 !important; }
        .class-progress-track, .class-overall-track { background: #34445A !important; }
        .class-overall-card {
            border-color: rgba(226,182,108,.48) !important;
            background: linear-gradient(115deg, rgba(232,189,115,.16), rgba(21,34,53,.96)) !important;
        }

        
        /* FINAL MODEL SELECTOR OVERRIDE: champagne-gold control */
        .st-key-model_picker_panel [data-testid="stSelectbox"] > label {
            color: #F0CC8B !important;
            -webkit-text-fill-color: #F0CC8B !important;
            font-weight: 800 !important;
        }
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] {
            background: linear-gradient(115deg, #F2D293 0%, #E8BD73 52%, #D9A45B 100%) !important;
            background-color: #E8BD73 !important;
            border: 1.5px solid #F8DDAA !important;
            border-radius: 14px !important;
            box-shadow:
                0 7px 20px rgba(0, 0, 0, .22),
                0 0 0 3px rgba(232, 189, 115, .14),
                inset 0 1px 0 rgba(255, 255, 255, .42) !important;
        }
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"]:hover {
            background: linear-gradient(115deg, #F7DDA5 0%, #EFC77F 52%, #E3B36B 100%) !important;
            border-color: #FFF0CE !important;
        }
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] *,
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] input {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            font-weight: 850 !important;
        }
        .st-key-model_picker_panel [data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: #172033 !important;
            fill: #172033 !important;
            opacity: 1 !important;
        }

    /* Model-comparison action: focused champagne button, no red primary ring */
    .st-key-run_model_comparison button,
    .st-key-run_model_comparison button:hover,
    .st-key-run_model_comparison button:focus,
    .st-key-run_model_comparison button:focus-visible,
    .st-key-run_model_comparison button:active,
    .st-key-run_model_comparison [data-testid="stBaseButton-secondary"] {
        min-height: 48px !important;
        padding: .72rem 1.35rem !important;
        border: 1px solid rgba(255, 231, 187, .78) !important;
        border-radius: 13px !important;
        background: linear-gradient(135deg, #F0CC8B 0%, #D9A95E 100%) !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: .9rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        letter-spacing: -.01em !important;
        box-shadow:
            0 7px 18px rgba(226, 182, 108, .16),
            inset 0 1px 0 rgba(255, 255, 255, .34) !important;
        outline: none !important;
        transition:
            background .18s ease,
            border-color .18s ease,
            box-shadow .18s ease,
            transform .18s ease !important;
    }
    .st-key-run_model_comparison button p,
    .st-key-run_model_comparison button span,
    .st-key-run_model_comparison button div {
        margin: 0 !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    .st-key-run_model_comparison button:hover {
        background: linear-gradient(135deg, #F7DCA8 0%, #E8BD73 100%) !important;
        border-color: #FFF0CF !important;
        box-shadow:
            0 10px 24px rgba(226, 182, 108, .23),
            inset 0 1px 0 rgba(255, 255, 255, .42) !important;
        transform: translateY(-1px);
    }
    .st-key-run_model_comparison button:focus,
    .st-key-run_model_comparison button:focus-visible,
    .st-key-run_model_comparison button:active {
        outline: none !important;
        border-color: #FFF0CF !important;
        box-shadow:
            0 0 0 2px rgba(232, 189, 115, .20),
            0 7px 18px rgba(226, 182, 108, .16),
            inset 0 1px 0 rgba(255, 255, 255, .34) !important;
    }
    .st-key-run_model_comparison button:disabled {
        opacity: .62 !important;
        cursor: not-allowed !important;
        transform: none !important;
    }

    </style>
        """,
        unsafe_allow_html=True,
    )

    plt.rcParams.update({
        "figure.facecolor": "#111a29",
        "axes.facecolor": "#111a29",
        "axes.edgecolor": "#53647a",
        "axes.labelcolor": "#d5deea",
        "text.color": "#edf2fa",
        "xtick.color": "#aab6c8",
        "ytick.color": "#aab6c8",
        "grid.color": "#34445a",
    })


# MODEL AND METADATA LOADING

@st.cache_data(show_spinner=False)
def load_metadata():
    """Load only the small metadata file; do not load model objects at startup."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "model_metadata.json was not found. Export the trained models "
            "and metadata from your notebook into the app folder."
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "models" not in metadata or "features" not in metadata:
        raise ValueError(
            "model_metadata.json must contain 'models' and 'features'."
        )
    if not metadata["models"] or not metadata["features"]:
        raise ValueError(
            "model_metadata.json must list at least one model and one feature."
        )

    return metadata


@st.cache_resource(show_spinner=False)
def load_model(file_stem, model_mtime_ns):
    """Load and cache one classifier when it is first needed."""
    model_path = MODEL_DIR / f"{file_stem}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load all classifiers only when the user explicitly runs model comparison."""
    metadata = load_metadata()
    loaded_models = {}

    for file_stem, display_name in metadata["models"].items():
        model_path = MODEL_DIR / f"{file_stem}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model file not found: {model_path}")
        loaded_models[display_name] = load_model(
            file_stem,
            model_path.stat().st_mtime_ns,
        )

    return loaded_models



# DATA VALIDATION AND EVALUATION
def load_test_data(uploaded_file, feature_columns):
    # UploadedFile behaves like a file handle. Reading it directly can fail
    # when its cursor has already moved to the end during a previous rerun.
    # getvalue() always returns the complete uploaded file contents.
    file_bytes = uploaded_file.getvalue()

    if not file_bytes or not file_bytes.strip():
        raise ValueError(
            "The uploaded CSV is empty. Please choose the exported test_data.csv "
            "file (not an empty placeholder)."
        )

    try:
        data = pd.read_csv(io.BytesIO(file_bytes))
    except pd.errors.EmptyDataError:
        raise ValueError(
            "Pandas could not find a CSV header or data rows. The selected file "
            "may be empty, contain only blank lines, or not be the test_data.csv "
            "exported from your notebook."
        )
    except UnicodeDecodeError:
        raise ValueError(
            "The CSV encoding could not be read. Save the file as UTF-8 CSV "
            "and upload it again."
        )

    if data.empty:
        raise ValueError("The uploaded CSV contains a header but no data records.")

    if len(data) < 500:
        raise ValueError(
            f"The assignment requires at least 500 test instances; "
            f"the uploaded dataset contains {len(data):,}."
        )

    if len(feature_columns) < 12:
        raise ValueError(
            f"The assignment requires at least 12 input features; "
            f"the selected model expects {len(feature_columns):,}."
        )

    required_columns = list(feature_columns) + ["target"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    raw_features = data[list(feature_columns)].copy()

    if raw_features.isna().any().any():
        bad_columns = raw_features.columns[raw_features.isna().any()].tolist()
        raise ValueError(
            "Missing values found in required feature columns: "
            + ", ".join(bad_columns)
            + ". Please clean the test dataset and upload it again."
        )

    X = raw_features.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Non-numeric values found in required feature columns: "
            + ", ".join(bad_columns)
            + ". Please upload numeric feature values."
        )

    target_raw = data["target"]
    if target_raw.isna().any():
        raise ValueError("The target column contains missing values.")

    y = pd.to_numeric(target_raw, errors="coerce")
    if y.isna().any():
        raise ValueError("The target column contains non-numeric values.")

    if not set(y.unique()).issubset({-1, 1}):
        raise ValueError("Target values must be -1 (Legitimate) or 1 (Phishing).")

    if y.nunique() != 2:
        raise ValueError(
            "The test dataset must contain both classes: -1 and 1."
        )

    return data, X, y.astype(int)


def get_model_scores(model, X):
    """Return a score for the positive class (1), for ROC/PR curves."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The selected model does not contain class 1.")

    positive_index = classes.index(1)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, positive_index]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if len(classes) == 2 and positive_index == 0:
            scores = -scores
        return scores

    raise ValueError(
        "This model supports neither predict_proba nor decision_function."
    )


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    scores = get_model_scores(model, X)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(y, predictions, pos_label=1, zero_division=0),
        "F1-score": f1_score(y, predictions, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    return predictions, scores, metrics


def plot_confusion_matrix(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[-1, 1])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        cbar=False,
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix", pad=14, weight="bold")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    auc_value = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"Model (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, linewidth=2.5)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig



# TOP MASTHEAD AND BITS IDENTITY
brand_col = st.container()

with brand_col:
    logo_col, title_col = st.columns([0.11, 0.89], vertical_alignment="center")
    with logo_col:
        # Official BITS Pilani colour emblem hosted on the institute website.
        st.image(
            "https://www.bits-pilani.ac.in/wp-content/uploads/image-44.png",
            width=66,
        )
    with title_col:
        st.markdown(
            """
            <div class="mast-brand">BITS PILANI</div>
            <div class="mast-meta" style="text-align:left;">
                MACHINE LEARNING · ASSIGNMENT 02
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="student-line">
        <span>STUDENT:  <strong>Harshil Manoj</strong></span>
        <span>STUDENT ID:  <strong>2026AJ05235</strong></span>
        <span>PROGRAMME:  <strong>M.Tech Artificial Intelligence &amp; Machine Learning</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <nav class="glass-nav" aria-label="Page sections">
        <a href="#overview" class="nav-link active">Overview</a>
        <a href="#dataset-upload" class="nav-link">Upload Data</a>
        <a href="#model-evaluation" class="nav-link">Model Evaluation</a>
        <a href="#model-comparison" class="nav-link">Model Comparison</a>
        <a href="#dataset-explorer" class="nav-link">Dataset Explorer</a>
    </nav>
    """,
    unsafe_allow_html=True,
)


def render_upload():
    st.markdown(
        """
        <div class="upload-panel">
            <div class="upload-kicker">DATA WORKSPACE · TEST SET</div>
            <div class="upload-title">Upload your labelled test dataset</div>
            <div class="upload-copy">
                Select the CSV exported from your notebook. Once uploaded, it is shared
                across model evaluation, comparison and dataset exploration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose test_data.csv",
        type=["csv"],
        key="main_test_upload",
        help="The CSV must contain the model features and a target column.",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        uploaded_size = len(uploaded.getvalue())
        if uploaded_size == 0:
            st.error("The selected file is empty (0 bytes). Please upload your exported test CSV.")
        else:
            st.success(
                f"Dataset selected · {uploaded.name} · {uploaded_size / 1024:.1f} KB"
            )
    else:
        st.caption("No test dataset uploaded. Add your CSV here to enable the analysis pages.")
    return uploaded



# REUSABLE PAGE HEADER
def page_header(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="section-note">{description}</div>', unsafe_allow_html=True)


# Load lightweight metadata only. Classifier files are loaded on demand.
try:
    metadata = load_metadata()
    feature_columns = metadata["features"]
    metadata_error = None
except Exception as error:
    metadata, feature_columns = {}, []
    metadata_error = str(error)



# OVERVIEW / HOME PAGE
st.markdown('<div id="overview" class="scroll-anchor"></div>', unsafe_allow_html=True)
if True:
    st.markdown(
        """
        <section class="home-hero">
            <div>
                <div class="home-kicker">MACHINE LEARNING · CLASSIFICATION PROJECT</div>
                <div class="home-title">Phishing website<br><span>detection.</span></div>
                <p class="home-copy">
                    A model evaluation application that studies how machine learning
                    algorithms distinguish legitimate websites from phishing websites.
                    Explore the test data, evaluate individual classifiers and compare
                    their performance through clear metrics and visual analysis.
                </p>
            </div>
            <div class="hero-side">
                <div class="hero-side-label">HOW THE APPLICATION WORKS</div>
                <div class="flow-step"><span class="flow-number">01</span> Upload labelled test data</div>
                <div class="flow-step"><span class="flow-number">02</span> Evaluate trained classifiers</div>
                <div class="flow-step"><span class="flow-number">03</span> Compare model performance</div>
                <div class="flow-step"><span class="flow-number">04</span> Explore results and patterns</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-student">
            <div class="student-item">
                <span class="student-label">Student name</span>
                <span class="student-value">Harshil Manoj</span>
            </div>
            <div class="student-item">
                <span class="student-label">Student ID</span>
                <span class="student-value">2026AJ05235</span>
            </div>
            <div class="student-item">
                <span class="student-label">Programme</span>
                <span class="student-value">M.Tech · Artificial Intelligence &amp; Machine Learning</span>
            </div>
            <div class="student-item">
                <span class="student-label">Assessment</span>
                <span class="student-value">Assignment 02</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">THE WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("## Everything you need to evaluate classification models")
    st.markdown(
        '<div class="section-intro">Move between four focused areas. Each section has a clear purpose, '
        'so you can go from raw test data to model insights without getting lost.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
            <a class="feature-card feature-link" href="#dataset-explorer">
                <div class="feature-icon">⇧</div>
                <div class="feature-title">Dataset Explorer <span class="feature-arrow">↗</span></div>
                <div class="feature-copy">Inspect test records, feature distributions, class balance and dataset quality.</div>
            </a>
            <a class="feature-card feature-link" href="#model-evaluation">
                <div class="feature-icon">◎</div>
                <div class="feature-title">Model Evaluation <span class="feature-arrow">↗</span></div>
                <div class="feature-copy">Evaluate a classifier using accuracy, precision, recall, F1, ROC-AUC and MCC.</div>
            </a>
            <a class="feature-card feature-link" href="#model-comparison">
                <div class="feature-icon">⇄</div>
                <div class="feature-title">Model Comparison <span class="feature-arrow">↗</span></div>
                <div class="feature-copy">Compare trained classifiers on the same test set and review metric-wise rankings.</div>
            </a>
            <a class="feature-card feature-link" href="#model-evaluation">
                <div class="feature-icon">▦</div>
                <div class="feature-title">Visual Diagnostics <span class="feature-arrow">↗</span></div>
                <div class="feature-copy">Explore confusion matrices, ROC curves and precision-recall curves.</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">CLASSIFIERS IN THIS PROJECT</div>', unsafe_allow_html=True)
    st.markdown("## Five approaches. One evaluation dataset.")
    st.markdown(
        '<div class="section-intro">The application is configured to load the trained models saved from your notebook.</div>',
        unsafe_allow_html=True,
    )

    model_names = list(metadata["models"].values()) if metadata else [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Gaussian Naive Bayes",
        "Support Vector Machine",
    ]
    model_cols = st.columns(min(5, max(1, len(model_names))))
    for index, model_name in enumerate(model_names):
        with model_cols[index % len(model_cols)]:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-no">CLASSIFIER {index + 1:02d}</div>
                    <div class="workflow-title">{model_name}</div>
                    <div class="workflow-copy">Trained classification model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-kicker">GET STARTED</div>', unsafe_allow_html=True)
    st.markdown("## A simple evaluation workflow")
    st.markdown(
        """
        <div class="workflow">
            <a class="workflow-card workflow-link" href="#dataset-upload">
                <div class="workflow-no">STEP 01 · ↗</div>
                <div class="workflow-title">Upload the test CSV</div>
                <div class="workflow-copy">Jump to the upload area and add your labelled test dataset.</div>
            </a>
            <a class="workflow-card workflow-link" href="#model-evaluation">
                <div class="workflow-no">STEP 02 · ↗</div>
                <div class="workflow-title">Choose what to inspect</div>
                <div class="workflow-copy">Jump to model evaluation or compare all available classifiers.</div>
            </a>
            <a class="workflow-card workflow-link" href="#model-comparison">
                <div class="workflow-no">STEP 03 · ↗</div>
                <div class="workflow-title">Interpret the results</div>
                <div class="workflow-copy">Review metrics, rankings and diagnostic plots to understand model performance.</div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div id="dataset-upload" class="scroll-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">YOUR DATASET</div>', unsafe_allow_html=True)
    st.markdown("## Bring your test data into the workspace")
    uploaded_file = render_upload()

    # Validate the uploaded test data once for all analysis sections.
    test_data = X_test = y_test = None
    data_error = None

    if uploaded_file is not None and not metadata_error:
        try:
            test_data, X_test, y_test = load_test_data(uploaded_file, feature_columns)
        except Exception as error:
            data_error = str(error)

    if metadata_error:
        with st.expander("Model setup information"):
            st.warning("Model metadata is not currently available.")
            st.code(metadata_error)
            st.caption(
                "Place model_metadata.json and the model folder beside app.py. "
                "The metadata must list the saved model filenames and feature columns."
            )

    if uploaded_file is not None and not metadata_error:
        if data_error:
            st.error(f"Dataset validation failed: {data_error}")
        elif test_data is not None:
            # Compact validation summary: only checks required by the assignment
            # and checks necessary to safely evaluate the selected classifier.
            with st.container(border=True):
                st.markdown("### Test dataset validation")
                st.success("Validation passed — the test dataset is ready for evaluation.")

                records_col, features_col, classes_col = st.columns(3)
                records_col.metric("Test records", f"{len(test_data):,}")
                features_col.metric("Input features", f"{len(feature_columns):,}")
                classes_col.metric("Target classes", f"{y_test.nunique()}")

                st.markdown("**Checks passed**")
                check_items = [
                    (
                        "Dataset size",
                        f"{len(test_data):,} records uploaded; minimum required is 500",
                    ),
                    (
                        "Feature size",
                        f"{len(feature_columns):,} model input features; minimum required is 12",
                    ),
                    (
                        "CSV structure",
                        "All required model features and the target column are present",
                    ),
                    (
                        "Data completeness",
                        "Required feature and target fields contain no missing values",
                    ),
                    (
                        "Target labels",
                        "Only -1 (Legitimate) and 1 (Phishing); both classes are present",
                    ),
                ]

                from html import escape

                checks_html = "".join(
                    (
                        '<div class="validation-check-item">'
                        '<span class="validation-check-icon" aria-label="Passed">✓</span>'
                        '<div class="validation-check-copy">'
                        f'<span class="validation-check-name">{escape(check_name)}</span>'
                        f'<span class="validation-check-detail">{escape(check_detail)}</span>'
                        '</div>'
                        '</div>'
                    )
                    for check_name, check_detail in check_items
                )
                checklist_html = (
                    '<div class="validation-check-grid">'
                    + checks_html
                    + '</div>'
                )

                # Use Streamlit's HTML renderer so the green status icons and
                # compact checklist cards render as UI, not as literal markup.
                if hasattr(st, "html"):
                    st.html(checklist_html)
                else:
                    left_checks, right_checks = st.columns(2, gap="large")
                    for index, (check_name, check_detail) in enumerate(check_items):
                        target_col = left_checks if index % 2 == 0 else right_checks
                        target_col.success(f"**{check_name}** — {check_detail}")



# MODEL EVALUATION PAGE
st.markdown('<div id="model-evaluation" class="scroll-anchor"></div>', unsafe_allow_html=True)
if True:
    st.markdown("---")
    page_header(
        "MODEL ANALYSIS",
        "Evaluate a classifier",
        "Select a trained model and inspect its predictive performance on the uploaded test dataset.",
    )

    if metadata_error:
        st.error("Model metadata is unavailable. Resolve the setup issue first.")
        st.code(metadata_error)
    elif uploaded_file is None:
        st.warning("No dataset uploaded. Use the dataset upload area above to select test_data.csv.")
    elif data_error:
        st.error(f"Dataset could not be loaded: {data_error}")
    else:
        top1, top2, top3 = st.columns(3)
        top1.metric("Test records", f"{len(test_data):,}")
        top2.metric("Input features", len(feature_columns))
        top3.metric("Available classifiers", len(metadata["models"]))

        # Dedicated, high-contrast glass panel makes model selection obvious.
        with st.container(key="model_picker_panel", border=True):
            picker_intro, picker_meta = st.columns(
                [0.70, 0.30], vertical_alignment="center"
            )
            with picker_intro:
                st.markdown(
                    """
                    <div class="model-picker-heading">Choose a classifier</div>
                    <div class="model-picker-copy">
                        Select a trained model to view its predictions, evaluation
                        metrics and diagnostic plots on your test dataset.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                selected_name = st.selectbox(
                    "SELECT CLASSIFICATION MODEL",
                    list(metadata["models"].values()),
                    label_visibility="visible",
                    key="evaluation_model_selector",
                )
            with picker_meta:
                st.markdown(
                    f"""
                    <div class="model-picker-meta">
                        <div class="model-picker-meta-label">AVAILABLE MODELS</div>
                        <div class="model-picker-meta-value">{len(metadata["models"])} trained classifiers</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        try:
            selected_stem = next(
                stem
                for stem, display_name in metadata["models"].items()
                if display_name == selected_name
            )
            selected_model_path = MODEL_DIR / f"{selected_stem}.joblib"

            with st.spinner(f"Loading {selected_name} and evaluating the test set..."):
                selected_model = load_model(
                    selected_stem,
                    selected_model_path.stat().st_mtime_ns,
                )
                predictions, scores, metrics = evaluate_model(
                    selected_model, X_test, y_test
                )

            st.markdown("### Performance metrics")
            metric_cols = st.columns(3)
            for index, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_cols[index % 3].metric(metric_name, f"{metric_value:.4f}")

            st.markdown("### Visual performance analysis")
            chart1, chart2 = st.columns(2)

            with chart1:
                st.pyplot(plot_confusion_matrix(y_test, predictions), use_container_width=True)
            with chart2:
                st.pyplot(plot_roc_curve(y_test, scores), use_container_width=True)

            chart3, chart4 = st.columns([1.05, 0.95], gap="large")
            with chart3:
                st.pyplot(
                    plot_precision_recall_curve(y_test, scores),
                    use_container_width=True,
                )

            with chart4:
                report = classification_report(
                    y_test,
                    predictions,
                    labels=[-1, 1],
                    target_names=["Legitimate", "Phishing"],
                    output_dict=True,
                    zero_division=0,
                )

                # Classification report: per-class metrics and average rows.
                report_table = pd.DataFrame(
                    [
                        {
                            "Class": "Legitimate",
                            "Precision": report["Legitimate"]["precision"],
                            "Recall": report["Legitimate"]["recall"],
                            "F1-score": report["Legitimate"]["f1-score"],
                            "Support": int(report["Legitimate"]["support"]),
                        },
                        {
                            "Class": "Phishing",
                            "Precision": report["Phishing"]["precision"],
                            "Recall": report["Phishing"]["recall"],
                            "F1-score": report["Phishing"]["f1-score"],
                            "Support": int(report["Phishing"]["support"]),
                        },
                        {
                            "Class": "Macro avg",
                            "Precision": report["macro avg"]["precision"],
                            "Recall": report["macro avg"]["recall"],
                            "F1-score": report["macro avg"]["f1-score"],
                            "Support": int(report["macro avg"]["support"]),
                        },
                        {
                            "Class": "Weighted avg",
                            "Precision": report["weighted avg"]["precision"],
                            "Recall": report["weighted avg"]["recall"],
                            "F1-score": report["weighted avg"]["f1-score"],
                            "Support": int(report["weighted avg"]["support"]),
                        },
                    ]
                )

                if DARK_MODE:
                    report_bg = "#111C2D"
                    report_fg = "#EAF0F7"
                    report_header_bg = "#25364D"
                    report_header_fg = "#F0CC8B"
                    report_border = "#35445A"
                    average_bg = "#1A2A40"
                    class_fg = "#F2D49C"
                else:
                    report_bg = "#FFFFFF"
                    report_fg = "#24344A"
                    report_header_bg = "#F2E7D2"
                    report_header_fg = "#77572C"
                    report_border = "#E2D6C1"
                    average_bg = "#F8F3E9"
                    class_fg = "#8A642F"

                table_rows_html = ""
                for row_index, row in report_table.iterrows():
                    row_class = "average-row" if row["Class"] in {"Macro avg", "Weighted avg"} else "class-row"
                    table_rows_html += (
                        f'<tr class="{row_class}">'
                        f'<th scope="row">{row["Class"]}</th>'
                        f'<td>{row["Precision"]:.2%}</td>'
                        f'<td>{row["Recall"]:.2%}</td>'
                        f'<td>{row["F1-score"]:.2%}</td>'
                        f'<td>{int(row["Support"]):,}</td>'
                        '</tr>'
                    )

                with st.container(key="classification_report_table", border=True):
                    st.markdown(
                        f"""
                        <div class="classification-table-scroll">
                            <table class="classification-report-table">
                                <thead>
                                    <tr>
                                        <th>Class</th>
                                        <th>Precision</th>
                                        <th>Recall</th>
                                        <th>F1-score</th>
                                        <th>Support</th>
                                    </tr>
                                </thead>
                                <tbody>{table_rows_html}</tbody>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Class-wise precision cards with the prediction counts behind each value.
                class_cm = confusion_matrix(y_test, predictions, labels=[-1, 1])

                tp_legitimate = int(class_cm[0, 0])
                fp_legitimate = int(class_cm[1, 0])
                tp_phishing = int(class_cm[1, 1])
                fp_phishing = int(class_cm[0, 1])

                legitimate_precision = float(report["Legitimate"]["precision"])
                phishing_precision = float(report["Phishing"]["precision"])

                legitimate_predicted_total = tp_legitimate + fp_legitimate
                phishing_predicted_total = tp_phishing + fp_phishing

                st.markdown(
                    '<div class="precision-stat-grid">'
                    '<div class="precision-stat-card legitimate">'
                    '<div class="precision-stat-top">'
                    '<span class="precision-stat-class">Legitimate</span>'
                    '<span class="precision-stat-tag">CLASS PRECISION</span>'
                    '</div>'
                    f'<div class="precision-stat-value">{legitimate_precision:.2%}</div>'
                    '<div class="precision-stat-detail">'
                    f'<strong>{tp_legitimate:,}</strong> correct out of '
                    f'<strong>{legitimate_predicted_total:,}</strong> predicted as Legitimate'
                    '</div>'
                    '<div class="precision-stat-track">'
                    f'<span style="width:{legitimate_precision * 100:.2f}%"></span>'
                    '</div>'
                    '</div>'
                    '<div class="precision-stat-card phishing">'
                    '<div class="precision-stat-top">'
                    '<span class="precision-stat-class">Phishing</span>'
                    '<span class="precision-stat-tag">CLASS PRECISION</span>'
                    '</div>'
                    f'<div class="precision-stat-value">{phishing_precision:.2%}</div>'
                    '<div class="precision-stat-detail">'
                    f'<strong>{tp_phishing:,}</strong> correct out of '
                    f'<strong>{phishing_predicted_total:,}</strong> predicted as Phishing'
                    '</div>'
                    '<div class="precision-stat-track">'
                    f'<span style="width:{phishing_precision * 100:.2f}%"></span>'
                    '</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Precision tells you how many records predicted as a class were actually correct."
                )

            with st.expander("View test data and predictions"):
                results = X_test.copy()
                results["Actual"] = y_test.to_numpy()
                results["Predicted"] = predictions
                results["Correct"] = results["Actual"] == results["Predicted"]
                st.dataframe(results.head(100), use_container_width=True)

                st.download_button(
                    "Download prediction results",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="phishing_predictions.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(f"Could not evaluate this model: {error}")



# MODEL COMPARISON PAGE

st.markdown('<div id="model-comparison" class="scroll-anchor"></div>', unsafe_allow_html=True)
if True:
    st.markdown("---")
    page_header(
        "MODEL BENCHMARKING",
        "Compare all classifiers",
        "Every model is evaluated on the same uploaded test records for a fair comparison.",
    )

    if metadata_error:
        st.error("Model metadata is unavailable.")
        st.code(metadata_error)
    elif uploaded_file is None:
        st.warning("Upload test_data.csv in the dataset section near the top of this page.")
    elif data_error:
        st.error(f"Dataset could not be loaded: {data_error}")
    else:
        # Keep comparison results for this exact uploaded CSV. A new CSV
        # clears the old comparison so results cannot be confused across datasets.
        dataset_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("comparison_dataset_signature") != dataset_signature:
            st.session_state["comparison_dataset_signature"] = dataset_signature
            st.session_state["comparison_results"] = None
            st.session_state["comparison_errors"] = []
            st.session_state["comparison_load_error"] = None

        st.markdown(
            "Run the benchmark when you are ready. The five saved classifiers "
            "will be loaded and evaluated once, then the results will remain available."
        )

        if st.button(
            "Run comparison across all classifiers",
            type="secondary",
            key="run_model_comparison",
            use_container_width=False,
        ):
            comparison_rows = []
            failed_models = []

            try:
                with st.spinner(
                    "Loading the saved classifiers and comparing model performance..."
                ):
                    comparison_models = load_models()

                    for model_name, model in comparison_models.items():
                        try:
                            _, _, model_metrics = evaluate_model(model, X_test, y_test)
                            comparison_rows.append(
                                {"Model": model_name, **model_metrics}
                            )
                        except Exception as error:
                            failed_models.append((model_name, str(error)))

                if comparison_rows:
                    st.session_state["comparison_results"] = (
                        pd.DataFrame(comparison_rows).set_index("Model")
                    )
                else:
                    st.session_state["comparison_results"] = None

                st.session_state["comparison_errors"] = failed_models
                st.session_state["comparison_load_error"] = None

            except Exception as error:
                st.session_state["comparison_results"] = None
                st.session_state["comparison_errors"] = []
                st.session_state["comparison_load_error"] = str(error)

        comparison_error = st.session_state.get("comparison_load_error")
        if comparison_error:
            st.error(f"Could not load the classifiers for comparison: {comparison_error}")

        comparison = st.session_state.get("comparison_results")
        failed_models = st.session_state.get("comparison_errors", [])

        if comparison is not None:

            st.markdown("### Model performance summary")
            st.caption(
                "Gold-highlighted cells identify the best-performing model for each metric."
            )

            # Highlight the highest score in every metric column.
            # This is data-driven, so winners update automatically when the
            # uploaded test dataset or evaluated model scores change.
            winner_style = (
                comparison.style
                .format("{:.4f}")
                .highlight_max(
                    axis=0,
                    props=(
                        "background-color: #3A3020; "
                        "color: #F7D58D; "
                        "font-weight: 800; "
                        "border: 1px solid #D6A95D;"
                    ),
                )
            )
            st.dataframe(
                winner_style,
                use_container_width=True,
            )

            st.markdown("### Metric comparison")
            st.caption(
                "Compare each classifier side by side. Longer bars indicate stronger "
                "performance; values are shown directly for easier interpretation."
            )
            metric_options = [
                "Accuracy", "ROC-AUC", "Precision", "Recall", "F1-score", "MCC"
            ]
            selected_metrics = st.multiselect(
                "Choose metrics to compare",
                metric_options,
                default=["Accuracy", "Precision", "Recall", "F1-score"],
                key="comparison_metrics_selector",
            )

            if selected_metrics:
                # Grouped horizontal bars keep metrics independent; unlike a
                # stacked chart, their values are never added together.
                from matplotlib.ticker import PercentFormatter

                plot_metrics = [m for m in selected_metrics if m != "MCC"]
                has_mcc = "MCC" in selected_metrics

                metric_palette = {
                    "Accuracy": "#E8BD73",   # signature champagne gold
                    "ROC-AUC": "#91AAA4",    # muted sage
                    "Precision": "#CDBB9A",  # warm champagne
                    "Recall": "#A98278",     # muted bronze-rose
                    "F1-score": "#91A4BE",   # slate blue
                }

                # Rank models by F1-score, then MCC, so the chart has a
                # consistent and meaningful order.
                ordered_models = comparison.sort_values(
                    ["F1-score", "MCC"], ascending=False
                ).index.tolist()

                if plot_metrics:
                    fig_height = max(4.2, 1.05 * len(ordered_models) + 1.8)
                    fig, ax = plt.subplots(figsize=(12.5, fig_height), dpi=120)
                    if DARK_MODE:
                        fig.patch.set_facecolor("#090E18")
                        ax.set_facecolor("#101C30")
                        foreground = "#EDF2FA"
                        muted = "#A7B3C5"
                        grid_color = "#2A394F"
                        title_color = "#F0CC8B"
                    else:
                        fig.patch.set_facecolor("#F6F7F9")
                        ax.set_facecolor("#FFFFFF")
                        foreground = "#24344A"
                        muted = "#718096"
                        grid_color = "#DCE3EB"
                        title_color = "#8A642F"

                    y_positions = list(range(len(ordered_models)))
                    group_height = 0.76
                    bar_height = group_height / max(len(plot_metrics), 1)

                    for metric_index, metric in enumerate(plot_metrics):
                        values = comparison.loc[ordered_models, metric].astype(float).tolist()
                        offsets = [
                            y - group_height / 2 + metric_index * bar_height
                            for y in y_positions
                        ]
                        bars = ax.barh(
                            offsets,
                            values,
                            height=bar_height * 0.86,
                            color=metric_palette[metric],
                            label=metric,
                            edgecolor="none",
                            zorder=3,
                        )

                        for bar, value in zip(bars, values):
                            ax.text(
                                min(value + 0.012, 1.015),
                                bar.get_y() + bar.get_height() / 2,
                                f"{value:.3f}",
                                va="center",
                                ha="left",
                                fontsize=9,
                                fontweight="bold",
                                color=foreground,
                                clip_on=False,
                            )

                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(ordered_models, fontsize=10, color=foreground)
                    ax.invert_yaxis()
                    ax.set_xlim(0, 1.12)
                    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
                    ax.set_xlabel("Score", color=muted, labelpad=10)
                    ax.set_title(
                        "Classification performance by model",
                        loc="left",
                        fontsize=15,
                        fontweight="bold",
                        color=title_color,
                        pad=18,
                    )
                    ax.xaxis.grid(True, color=grid_color, linewidth=0.8, alpha=0.9, zorder=0)
                    ax.set_axisbelow(True)
                    ax.tick_params(axis="x", colors=muted, labelsize=9, length=0)
                    ax.tick_params(axis="y", colors=foreground, labelsize=10, length=0, pad=10)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.13),
                        ncol=min(len(plot_metrics), 5),
                        frameon=False,
                        labelcolor=foreground,
                        fontsize=9,
                    )
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                if has_mcc:
                    st.markdown("#### Matthews Correlation Coefficient (MCC)")
                    st.caption(
                        "MCC uses a different scale from the other metrics: −1 is "
                        "worst, 0 is no better than random association, and +1 is perfect."
                    )
                    mcc_values = comparison.loc[ordered_models, "MCC"].astype(float)
                    fig_mcc, ax_mcc = plt.subplots(
                        figsize=(12.5, 2.7 + 0.45 * len(ordered_models)), dpi=120
                    )
                    if DARK_MODE:
                        fig_mcc.patch.set_facecolor("#090E18")
                        ax_mcc.set_facecolor("#101C30")
                        mcc_text = "#EDF2FA"
                        mcc_grid = "#2A394F"
                        mcc_title = "#F0CC8B"
                    else:
                        fig_mcc.patch.set_facecolor("#F6F7F9")
                        ax_mcc.set_facecolor("#FFFFFF")
                        mcc_text = "#24344A"
                        mcc_grid = "#DCE3EB"
                        mcc_title = "#8A642F"

                    mcc_colors = [
                        "#E8BD73" if value >= 0 else "#D98282"
                        for value in mcc_values
                    ]
                    bars = ax_mcc.barh(
                        ordered_models,
                        mcc_values.values,
                        color=mcc_colors,
                        height=0.58,
                        zorder=3,
                    )
                    ax_mcc.axvline(0, color=mcc_text, linewidth=1, alpha=.65)
                    for bar, value in zip(bars, mcc_values.values):
                        ax_mcc.text(
                            value + (0.035 if value >= 0 else -0.035),
                            bar.get_y() + bar.get_height() / 2,
                            f"{value:.3f}",
                            va="center",
                            ha="left" if value >= 0 else "right",
                            color=mcc_text,
                            fontsize=9,
                            fontweight="bold",
                        )
                    ax_mcc.set_xlim(-1.12, 1.12)
                    ax_mcc.set_title(
                        "MCC by classifier",
                        loc="left",
                        fontsize=14,
                        fontweight="bold",
                        color=mcc_title,
                        pad=12,
                    )
                    ax_mcc.xaxis.grid(True, color=mcc_grid, linewidth=.8, zorder=0)
                    ax_mcc.set_axisbelow(True)
                    ax_mcc.tick_params(axis="x", colors=mcc_text, labelsize=9, length=0)
                    ax_mcc.tick_params(axis="y", colors=mcc_text, labelsize=10, length=0)
                    for spine in ax_mcc.spines.values():
                        spine.set_visible(False)
                    fig_mcc.tight_layout()
                    st.pyplot(fig_mcc, use_container_width=True)
                    plt.close(fig_mcc)

                st.caption(
                    "Note: Metrics are displayed independently, not stacked. "
                    "The main chart uses a 0–100% scale; MCC is shown separately "
                    "because its valid range is −1 to +1."
                )
            else:
                st.info("Select at least one metric to display the comparison chart.")

            ranking = comparison.sort_values(
                ["F1-score", "MCC"], ascending=False
            )
            best_model = ranking.index[0]
            st.success(
                f"Top-ranked model by F1-score (MCC as tie-breaker): {best_model}"
            )

            st.markdown("### Model ranking")
            ranking_display = ranking.reset_index()
            ranking_display.insert(0, "Rank", range(1, len(ranking_display) + 1))
            st.dataframe(ranking_display, use_container_width=True)

            st.download_button(
                "Download comparison results",
                comparison.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="model_comparison_results.csv",
                mime="text/csv",
            )

        if comparison is None and not comparison_error:
            st.info(
                "Model comparison has not been run yet. Select **Run comparison "
                "across all classifiers** to load the five models and calculate "
                "their metrics."
            )

        for model_name, error_message in failed_models:
            st.warning(f"{model_name} could not be evaluated: {error_message}")



# DATASET EXPLORER PAGE
st.markdown('<div id="dataset-explorer" class="scroll-anchor"></div>', unsafe_allow_html=True)
if True:
    st.markdown("---")
    page_header(
        "DATA INSPECTION",
        "Explore your test dataset",
        "Inspect the uploaded records, feature distributions and target-class balance.",
    )

    if uploaded_file is None:
        st.warning("Upload test_data.csv in the dataset section near the top of this page.")
    elif data_error:
        st.error(f"Dataset could not be loaded: {data_error}")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows", f"{test_data.shape[0]:,}")
        d2.metric("Columns", test_data.shape[1])
        d3.metric("Missing values", int(test_data.isna().sum().sum()))
        d4.metric("Duplicate rows", int(test_data.duplicated().sum()))

        st.markdown("### Dataset preview")
        st.dataframe(test_data.head(20), use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.markdown("### Target class distribution")
            target_counts = y_test.map(
                {-1: "Legitimate", 1: "Phishing"}
            ).value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts.plot(
                kind="bar", ax=ax, rot=0, color=["#0b8f83", "#ed7891"]
            )
            ax.set_xlabel("Website class")
            ax.set_ylabel("Number of records")
            ax.set_title("Class distribution")
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("### Feature distribution")
            selected_feature = st.selectbox(
                "Choose a feature",
                list(feature_columns),
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(X_test[selected_feature], bins=25, edgecolor="white")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {selected_feature}")
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with st.expander("View descriptive statistics"):
            st.dataframe(X_test.describe().T, use_container_width=True)



# FOOTER

st.markdown(
    '<div class="footer">BITS PILANI &nbsp; · &nbsp; PHISHING WEBSITE DETECTION &nbsp; · &nbsp; MACHINE LEARNING ASSIGNMENT 02</div>',
    unsafe_allow_html=True,
)
