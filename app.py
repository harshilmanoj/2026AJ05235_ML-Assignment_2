import json
import io
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st


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

# The app uses one consistent dark theme.
DARK_MODE = True




# Lightweight styling: use Streamlit's system font stack; no external font/API requests.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.025em; }
.student-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem 1.8rem; padding:.85rem .2rem; margin-top:.2rem; margin-bottom:1.1rem; line-height:1.5; border-bottom:1px solid #263449; color:#93a2b8; font-size:.82rem; }
.student-strip strong { color:#e5ebf4; font-weight:700; }
.intro-panel {
    display:grid; grid-template-columns:minmax(0,1fr) minmax(300px,340px);
    gap:clamp(1.25rem,3vw,2.5rem); align-items:center;
    padding:clamp(1.25rem,2.5vw,1.8rem); margin-bottom:1.1rem;
    border:1px solid #2a3a50; border-radius:15px; background:#14243a;
}
.intro-panel > div:first-child { min-width:0; }
.intro-kicker { color:#7dd8c5; font-size:.7rem; font-weight:750; letter-spacing:.13em; }
.intro-title { color:#f3f6fb; font-size:clamp(1.8rem,3vw,2.5rem); line-height:1.12; font-weight:800; letter-spacing:-.04em; margin:.45rem 0 .65rem; }
.intro-copy { color:#aab8ca; font-size:clamp(.9rem,1.2vw,1rem); line-height:1.65; margin:0; max-width:760px; }
.intro-steps { padding:.9rem 1.05rem; border:1px solid #40516a; border-radius:12px; background:#1b2d43; }
.intro-steps-title { color:#a9bacb; font-size:.67rem; font-weight:750; letter-spacing:.12em; margin-bottom:.45rem; }
.intro-step { display:flex; align-items:flex-start; gap:.55rem; color:#e5ebf4; font-size:.82rem; line-height:1.4; padding:.48rem 0; border-top:1px solid #35465b; }
.step-number { flex:0 0 1.55rem; color:#8bdacb; font-weight:800; font-variant-numeric:tabular-nums; }
.intro-step-content { flex:1; min-width:0; }
.intro-options { display:flex; flex-wrap:wrap; gap:.3rem; margin:.38rem 0 .05rem; }
.intro-options span {
    display:inline-block; color:#b9c8d9; font-size:.68rem; line-height:1.3;
    padding:.22rem .42rem; border:1px solid #3b526b; border-radius:5px;
    background:#223850; white-space:nowrap;
}
@media(max-width:900px) {
    .intro-panel { grid-template-columns:1fr; gap:1.1rem; }
    .intro-steps { max-width:none; }
}
@media(max-width:560px) {
    .student-strip { gap:.35rem .9rem; font-size:.74rem; }
    .intro-panel { padding:1rem; }
    .intro-title { font-size:1.8rem; }
    .intro-copy { font-size:.88rem; }
    .intro-options span { white-space:normal; }
}
.section-card { background: #111c2d; border: 1px solid #2a3a50; border-radius: 12px; padding: 1rem 1.1rem; }
div[data-testid="stMetric"] { background: #111c2d; border: 1px solid #2a3a50; border-radius: 10px; padding: .8rem; }
.stButton button, .stDownloadButton button { border-radius: 8px; font-weight: 650; }
.st-key-model_picker_panel { background:#111c2d; border-color:#2a3a50 !important; border-radius:12px; padding:.4rem .5rem .6rem; }
.model-picker-heading { color:#f3f6fb; font-size:1.2rem; font-weight:700; letter-spacing:-.01em; margin:0 0 .2rem; }
.model-picker-copy { color:#93a2b8; font-size:.88rem; line-height:1.45; margin:0; }
.model-picker-meta { display:flex; flex-direction:column; align-items:flex-end; text-align:right; }
.model-picker-meta-label { color:#93a2b8; font-size:.66rem; font-weight:750; letter-spacing:.12em; }
.model-picker-meta-value { color:#f7d58d; font-size:1.05rem; font-weight:800; }
.st-key-model_picker_panel [data-testid="stSelectbox"] { width:100%; margin-top:.4rem; }
.st-key-model_picker_panel [data-testid="stWidgetLabel"] p {
    color:#e8bd73; font-size:.72rem; font-weight:800; letter-spacing:.1em;
}
.st-key-model_picker_panel [data-baseweb="select"] {
    background:#17283f; border:1.5px solid #d9ad68; border-radius:10px;
}
.st-key-model_picker_panel [data-baseweb="select"]:hover { border-color:#f0cc8b; }
.st-key-model_picker_panel [data-baseweb="select"] > div {
    background:transparent !important; border:0 !important; min-height:3rem;
}
.st-key-model_picker_panel [data-baseweb="select"] * { color:#f5e6c9 !important; font-weight:600; }
.st-key-model_picker_panel [data-baseweb="select"] svg { fill:#e8bd73 !important; color:#e8bd73 !important; }
[data-testid="stFileUploader"] section { background: #111c2d; border: 1px dashed #64748b; border-radius: 10px; }
[data-testid="stDataFrame"] { border: 1px solid #2a3a50; border-radius: 9px; }
/* classification report */

.st-key-classification_report_table {
    aspect-ratio: 6 / 4; display:flex; flex-direction:column; justify-content:center;
    background:#111a29; border-radius:10px; padding:.6rem 1rem; box-sizing:border-box;
}
.report-title { text-align:center; color:#edf2fa; font-size:1rem; font-weight:700; margin:0 0 .7rem; }
.classification-table-scroll { width:100%; overflow-x:auto; border-radius:10px; }
.classification-report-table {
    width:100%; table-layout:fixed; border-collapse:collapse;
    font-size:.78rem; background:#111c2d; border:1px solid #35445a;
}
.classification-report-table th, .classification-report-table td {
    padding:.5rem .4rem; border:1px solid #35445a;
    text-align:center; overflow-wrap:anywhere;
}
.classification-report-table thead th {
    background:#25364d; color:#f0cc8b; font-weight:800;
}
.classification-report-table tbody th {
    background:#17283f; color:#f2d49c; text-align:left; font-weight:700;
}
.classification-report-table tbody td { color:#eaf0f7; }
.classification-report-table tbody tr.average-row th,
.classification-report-table tbody tr.average-row td {
    background:#1b2a40; font-weight:750;
}

/* Pinned analysis switcher: three equal cards, same look as the metric cards */
header[data-testid="stHeader"] { background:#0b1220; }
div:has(> .st-key-analysis_nav) { position: sticky; top: 3.6rem; z-index: 999; }
.st-key-analysis_nav {
    background:#0b1220; padding:.35rem 0 .5rem;
    border-bottom:1px solid #2a3a50;
    box-shadow:0 10px 18px -10px rgba(0,0,0,.7);
}
.analysis-title { color:#fafafa; font-size:1.75rem; font-weight:600; line-height:1.25; letter-spacing:-0.025em; margin:0 0 .5rem; }
.st-key-analysis_nav button {
    width:100%; height:2.5rem; min-height:2.5rem; padding:0 .75rem; border-radius:8px;
    background:#111c2d; border:1px solid #2a3a50; color:#cbd7e6;
}
.st-key-analysis_nav button p { font-size:.92rem; font-weight:700; margin:0; }
.st-key-analysis_nav button:hover { border-color:#d9ad68; color:#f0dcae; }
.st-key-analysis_nav button[data-testid="stBaseButton-primary"],
.st-key-analysis_nav button[kind="primary"] {
    background:#3a3020; border:1px solid #d9ad68; color:#f7d58d;
}
</style>
""", unsafe_allow_html=True)



# MODEL AND METADATA LOADING

@st.cache_data(show_spinner=False)
def load_metadata():
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
    model_path = MODEL_DIR / f"{file_stem}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    import joblib
    return joblib.load(model_path)



# DATA VALIDATION AND EVALUATION
@st.cache_data(show_spinner=False, max_entries=3)
def load_test_data_cached(file_bytes, feature_columns):
    """
    Return the current two-item format: (X, y).

    Compatibility guard: an earlier app version returned (data, X, y).
    Normalise that legacy result so callers never fail with an unpacking error.
    """
    loaded = _load_test_data(file_bytes, feature_columns)

    if not isinstance(loaded, tuple):
        raise ValueError("Dataset loader returned an unexpected result.")

    if len(loaded) == 3:
        # Legacy format: original DataFrame, feature matrix, target vector.
        return loaded[1], loaded[2]

    if len(loaded) == 2:
        return loaded

    raise ValueError(
        f"Dataset loader returned {len(loaded)} values; expected 2."
    )


def _load_test_data(file_bytes, feature_columns):
    if not file_bytes or not file_bytes.strip():
        raise ValueError(
            "The uploaded CSV is empty. Please choose the exported test_data.csv "
            "file (not an empty placeholder)."
        )

    if len(feature_columns) < 12:
        raise ValueError(
            f"The assignment requires at least 12 input features; "
            f"the selected model expects {len(feature_columns):,}."
        )

    required_columns = list(feature_columns) + ["target"]

    # Read the header first, then parse only the columns required by the models.
    # This avoids loading unrelated CSV columns into memory.
    try:
        csv_header = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
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

    missing_columns = [
        column for column in required_columns if column not in csv_header.columns
    ]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    try:
        data = pd.read_csv(io.BytesIO(file_bytes), usecols=required_columns)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV contains a header but no data records.")
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

    raw_features = data.loc[:, list(feature_columns)]

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

    return X, y.astype(int)


def get_model_scores(model, X):
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


@st.cache_data(show_spinner=False, max_entries=8)
def evaluate_selected_cached(file_stem, model_mtime_ns, file_bytes, feature_columns):
    from sklearn.metrics import classification_report
    X, y = load_test_data_cached(file_bytes, feature_columns)
    model = load_model(file_stem, model_mtime_ns)
    predictions, scores, metrics = evaluate_model(model, X, y)
    report = classification_report(
        y,
        predictions,
        labels=[-1, 1],
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
        zero_division=0,
    )
    return predictions, scores, metrics, report


@st.cache_data(show_spinner=False, max_entries=8)
def comparison_cached(file_bytes, feature_columns, model_mtimes):
    X, y = load_test_data_cached(file_bytes, feature_columns)
    metadata = load_metadata()
    rows, failed = [], []
    for file_stem, display_name in metadata["models"].items():
        try:
            model = load_model(file_stem, dict(model_mtimes)[file_stem])
            _, _, model_metrics = evaluate_model(model, X, y)
            rows.append({"Model": display_name, **model_metrics})
        except Exception as error:
            failed.append((display_name, str(error)))
    return rows, failed


@st.cache_data(show_spinner=False, max_entries=4)
def prediction_csv_cached(file_bytes, feature_columns, file_stem, model_mtime_ns):
    """Build the complete prediction export only when explicitly requested."""
    X, y = load_test_data_cached(file_bytes, feature_columns)
    predictions, _, _, _ = evaluate_selected_cached(
        file_stem, model_mtime_ns, file_bytes, feature_columns
    )
    export_data = X.copy()
    export_data["Actual"] = y.to_numpy()
    export_data["Predicted"] = predictions
    export_data["Correct"] = export_data["Actual"].to_numpy() == predictions
    return export_data.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=4)
def describe_cached(file_bytes, feature_columns):
    X, _ = load_test_data_cached(file_bytes, feature_columns)
    return X.describe().T


def evaluate_model(model, X, y):
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
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
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true, predictions, labels=[-1, 1])
    fig, ax = plt.subplots(figsize=(6, 4.0), dpi=100)

    ax.imshow(matrix, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix", pad=14, weight="bold")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column, row, f"{matrix[row, column]:,}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if matrix[row, column] > threshold else "#172238",
            )

    ax.set_xticks([-0.5, 0.5], minor=True)
    ax.set_yticks([-0.5, 0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout(pad=1.15)
    return fig


def plot_roc_curve(y_true, scores):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    auc_value = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 4.0), dpi=100)
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
    fig.tight_layout(pad=1.15)
    return fig


def plot_precision_recall_curve(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)

    fig, ax = plt.subplots(figsize=(6, 4.0), dpi=100)
    ax.plot(recall, precision, linewidth=2.5)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=.2)
    fig.tight_layout(pad=1.15)
    return fig






# ------------------------------------------------------------
# APP HEADER + LIGHTWEIGHT UPLOAD WORKFLOW
# ------------------------------------------------------------
# Lightweight student identity strip and project introduction.
# Static HTML/CSS only: no image, font, API, or model loading is triggered here.
st.markdown(
    """
    <div class="student-strip">
        <span>STUDENT: <strong>Harshil Manoj</strong></span>
        <span>STUDENT ID: <strong>2026AJ05235</strong></span>
        <span>PROGRAMME: <strong>M.Tech Artificial Intelligence &amp; Machine Learning</strong></span>
    </div>
    <section class="intro-panel">
        <div>
            <div class="intro-kicker">MACHINE LEARNING · CLASSIFICATION PROJECT</div>
            <div class="intro-title">Phishing Website Detection</div>
            <p class="intro-copy">
                A model evaluation application that studies how machine learning
                algorithms distinguish legitimate websites from phishing websites.
                Upload labelled test data, evaluate individual classifiers, compare
                their performance and explore the results.
            </p>
        </div>
        <aside class="intro-steps">
            <div class="intro-steps-title">HOW THE APPLICATION WORKS</div>
            <div class="intro-step"><span class="step-number">01</span><div class="intro-step-content">Upload labelled test data</div></div>
            <div class="intro-step"><span class="step-number">02</span><div class="intro-step-content">Validate the dataset</div></div>
            <div class="intro-step">
                <span class="step-number">03</span>
                <div class="intro-step-content">
                    Choose one analysis
                    <div class="intro-options">
                        <span>Evaluate a classifier</span>
                        <span>Compare classifiers</span>
                        <span>Explore the dataset</span>
                    </div>
                </div>
            </div>
            <div class="intro-step"><span class="step-number">04</span><div class="intro-step-content">Review results and export predictions</div></div>
        </aside>
    </section>
    """,
    unsafe_allow_html=True,
)


def page_header(kicker, title, description):
    st.caption(kicker)
    st.subheader(title)
    st.caption(description)

with st.container(border=True):
    st.markdown("### 1. Upload test dataset")
    st.caption("Upload the labelled CSV exported from your notebook. Analysis sections become available after validation.")
    uploaded_file = st.file_uploader(
        "Choose test_data.csv",
        type=["csv"],
        key="main_test_upload",
        help="The CSV must contain the model features and a target column.",
    )

# Load only small metadata at startup. Model files are loaded only inside the selected feature.
try:
    metadata = load_metadata()
    feature_columns = metadata["features"]
    metadata_error = None
except Exception as error:
    metadata, feature_columns = {}, []
    metadata_error = str(error)

X_test = y_test = None
data_error = None
upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
feature_key = tuple(feature_columns)

if metadata_error:
    st.error("Model configuration could not be loaded.")
    st.caption(metadata_error)
elif uploaded_file is not None:
    if not upload_bytes:
        data_error = "The uploaded CSV is empty."
    else:
        try:
            X_test, y_test = load_test_data_cached(upload_bytes, feature_key)
        except Exception as error:
            data_error = str(error)

if uploaded_file is not None and not metadata_error:
    if data_error:
        st.error(f"Dataset validation failed: {data_error}")
    elif X_test is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test records", f"{len(X_test):,}")
        c2.metric("Input features", len(feature_columns))
        c3.metric("Target classes", y_test.nunique())
        st.success(f"Dataset ready · {uploaded_file.name}")

# Navigation is deliberately below upload and validation.
# It is always shown once a valid dataset is uploaded, so you can switch
# between analyses at any time (normal flow, so it never overlaps content).
if uploaded_file is not None and X_test is not None and not metadata_error and not data_error:
    st.markdown("---")
    ANALYSES = ["Model Evaluation", "Model Comparison", "Dataset Explorer"]
    if st.session_state.get("active_analysis") not in ANALYSES:
        st.session_state["active_analysis"] = ANALYSES[0]

    def _set_analysis(name):
        st.session_state["active_analysis"] = name

    with st.container(key="analysis_nav"):
        st.markdown('<div class="analysis-title">2. Choose an analysis</div>', unsafe_allow_html=True)
        nav_cols = st.columns(3, gap="small")
        for nav_col, name in zip(nav_cols, ANALYSES):
            nav_col.button(
                name,
                key=f"nav_{name}",
                type="primary" if st.session_state["active_analysis"] == name else "secondary",
                use_container_width=True,
                on_click=_set_analysis,
                args=(name,),
            )
    selected_page = st.session_state["active_analysis"]
else:
    selected_page = None
    st.info("Upload a valid test CSV to unlock Model Evaluation, Model Comparison and Dataset Explorer.")

if selected_page is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#111a29", "axes.facecolor": "#111a29",
                         "axes.edgecolor": "#53647a", "axes.labelcolor": "#d5deea",
                         "text.color": "#edf2fa", "xtick.color": "#aab6c8",
                         "ytick.color": "#aab6c8", "grid.color": "#34445a"})



if selected_page == "Model Evaluation":
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
        top1.metric("Test records", f"{len(X_test):,}")
        top2.metric("Input features", len(feature_columns))
        top3.metric("Available classifiers", len(metadata["models"]))

        # Dedicated, high-contrast glass panel makes model selection obvious.
        with st.container(key="model_picker_panel", border=True):
            picker_intro, picker_meta = st.columns(
                [0.74, 0.26], vertical_alignment="center"
            )
            with picker_intro:
                st.markdown(
                    """
                    <div class="model-picker-heading">Choose a classifier</div>
                    <div class="model-picker-copy">
                        Select a trained model to view its evaluation metrics and diagnostic plots.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with picker_meta:
                st.markdown(
                    f"""
                    <div class="model-picker-meta">
                        <div class="model-picker-meta-label">AVAILABLE MODELS</div>
                        <div class="model-picker-meta-value">{len(metadata["models"])} classifiers</div>
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

        try:
            selected_stem = next(
                stem
                for stem, display_name in metadata["models"].items()
                if display_name == selected_name
            )
            selected_model_path = MODEL_DIR / f"{selected_stem}.joblib"

            selected_mtime = selected_model_path.stat().st_mtime_ns

            with st.spinner(f"Loading {selected_name} and evaluating the test set..."):
                predictions, scores, metrics, report = evaluate_selected_cached(
                    selected_stem, selected_mtime, upload_bytes, feature_key
                )

            st.markdown("### Performance metrics")
            metric_cols = st.columns(3)
            for index, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_cols[index % 3].metric(metric_name, f"{metric_value:.4f}")

            st.markdown("### Visual performance analysis")
            chart1, chart2 = st.columns(2, gap="large")

            with chart1:
                fig_cm = plot_confusion_matrix(y_test, predictions)
                st.pyplot(fig_cm, use_container_width=True, clear_figure=False)
                plt.close(fig_cm)
            with chart2:
                fig_roc = plot_roc_curve(y_test, scores)
                st.pyplot(fig_roc, use_container_width=True, clear_figure=False)
                plt.close(fig_roc)

            chart3, chart4 = st.columns(2, gap="large")
            with chart3:
                fig_pr = plot_precision_recall_curve(y_test, scores)
                st.pyplot(fig_pr, use_container_width=True, clear_figure=False)
                plt.close(fig_pr)

            with chart4:
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

                with st.container(key="classification_report_table", border=False):
                    st.markdown('<div class="report-title">Classification report</div>', unsafe_allow_html=True)
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

            if st.checkbox(
                "Show test data and prediction preview",
                key="show_prediction_preview",
            ):
                preview = X_test.head(100).copy()
                preview["Actual"] = y_test.head(100).to_numpy()
                preview["Predicted"] = predictions[:100]
                preview["Correct"] = (
                    preview["Actual"].to_numpy() == preview["Predicted"].to_numpy()
                )
                st.dataframe(preview, use_container_width=True)
                st.caption("Preview limited to the first 100 records.")

                export_signature = (
                    selected_stem,
                    selected_mtime,
                    hashlib.sha256(upload_bytes).hexdigest(),
                )

                if st.button("Prepare full prediction CSV", key="prepare_prediction_csv"):
                    with st.spinner("Preparing prediction export..."):
                        st.session_state["prediction_csv_data"] = prediction_csv_cached(
                            upload_bytes, feature_key, selected_stem, selected_mtime
                        )
                        st.session_state["prediction_csv_signature"] = export_signature

                if (
                    st.session_state.get("prediction_csv_signature") == export_signature
                    and st.session_state.get("prediction_csv_data") is not None
                ):
                    st.download_button(
                        "Download prediction results",
                        st.session_state["prediction_csv_data"],
                        file_name="phishing_predictions.csv",
                        mime="text/csv",
                        key="download_prediction_csv",
                    )

        except Exception as error:
            st.error(f"Could not evaluate this model: {error}")

if selected_page == "Model Comparison":
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
        dataset_signature = hashlib.sha256(upload_bytes).hexdigest()
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
                    model_mtimes = tuple(
                        (stem, (MODEL_DIR / f"{stem}.joblib").stat().st_mtime_ns)
                        for stem in metadata["models"]
                    )
                    comparison_rows, failed_models = comparison_cached(
                        upload_bytes,
                        feature_key,
                        model_mtimes,
                    )

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
                    fig, ax = plt.subplots(figsize=(12.5, fig_height), dpi=100)
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
                        figsize=(12.5, 2.7 + 0.45 * len(ordered_models)), dpi=100
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

if selected_page == "Dataset Explorer":
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
        d1, d2 = st.columns(2)
        d1.metric("Rows", f"{X_test.shape[0]:,}")
        d2.metric("Columns", X_test.shape[1] + 1)

        st.markdown("### Dataset preview")
        dataset_preview = X_test.head(20).copy()
        dataset_preview["target"] = y_test.head(20).to_numpy()
        st.dataframe(dataset_preview, use_container_width=True)

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

        if st.checkbox("Run optional data-quality checks", key="run_data_quality"):
            missing_count = int(X_test.isna().sum().sum() + y_test.isna().sum())
            duplicate_count = int(
                pd.concat([X_test, y_test.rename("target")], axis=1).duplicated().sum()
            )
            q1, q2 = st.columns(2)
            q1.metric("Missing values", missing_count)
            q2.metric("Duplicate rows", duplicate_count)

        if st.checkbox("Show descriptive statistics", key="show_descriptive_stats"):
            st.dataframe(
                describe_cached(upload_bytes, feature_key),
                use_container_width=True,
            )

st.markdown(
    '<div style="text-align:center;color:#718096;border-top:1px solid #263449;padding-top:1rem;margin-top:2rem;font-size:.75rem;">BITS PILANI · PHISHING WEBSITE DETECTION · ASSIGNMENT 02</div>',
    unsafe_allow_html=True,
)