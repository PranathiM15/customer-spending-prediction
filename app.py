import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="SpendWise AI", page_icon="✦", layout="wide")

# ---------- visual identity ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 12% 0%, #172c43 0, #0a1220 34%, #080d17 75%); color:#edf4ff;}
.block-container {padding-top:1.6rem; max-width:1450px;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; color:#f4f8ff;}
.hero {padding:28px 32px; border:1px solid #29435d; border-radius:22px; background:linear-gradient(115deg,rgba(22,48,72,.95),rgba(15,25,43,.92)); margin-bottom:22px;}
.eyebrow {color:#5ce1c2; letter-spacing:2px; font-size:11px; font-weight:700;}
.hero-title {font-size:36px; font-weight:700; margin:4px 0 8px; color:#fff;}
.hero-sub {color:#a9bed2; font-size:14px;}
.metric {background:linear-gradient(145deg,#14283c,#101b2c); border:1px solid #29435d; border-radius:16px; padding:18px 20px;}
.metric-label {color:#91a9bf; font-size:12px; text-transform:uppercase; letter-spacing:1px;}
.metric-value {font-family:'Space Grotesk'; color:#f4f8ff; font-size:28px; font-weight:700; margin-top:5px;}
.section {color:#5ce1c2; font-size:12px; font-weight:700; letter-spacing:1.6px; margin:18px 0 6px;}
div[data-testid="stSidebar"] {background:#0d1928; border-right:1px solid #23394f;}
.stButton>button {background:linear-gradient(90deg,#28c7a5,#48a8f5); color:#071522; border:0; border-radius:10px; font-weight:700;}
.stDownloadButton>button {border-radius:10px;}
[data-testid="stDataFrame"] {border:1px solid #29435d; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">CUSTOMER INTELLIGENCE / MACHINE LEARNING</div>
  <div class="hero-title">✦ SpendWise <span style="color:#5ce1c2">AI</span></div>
  <div class="hero-sub">Understand customer behaviour. Segment spending potential. Make smarter business decisions.</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample():
    return pd.read_csv("data/sample_customers.csv")

def clean_frame(frame):
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data.drop_duplicates()
    # Normalize common blank markers
    data = data.replace(["", " ", "NA", "N/A", "null", "None", "?"], np.nan)
    return data

def create_target(data):
    lower_map = {str(c).lower().replace("_","").replace(" ",""): c for c in data.columns}
    target = None
    for key in ["spendingcategory", "spendinglevel", "spendingclass", "segment"]:
        if key in lower_map:
            target = lower_map[key]
            break
    if target:
        data[target] = data[target].astype(str).str.strip().str.title()
        return data, target, "Existing spending category column"
    score_col = None
    for key in ["spendingscore", "spendingscore1100", "score"]:
        if key in lower_map:
            score_col = lower_map[key]
            break
    if score_col:
        values = pd.to_numeric(data[score_col], errors="coerce")
        data["SpendingCategory"] = pd.cut(values, bins=[-np.inf,39,69,np.inf],
                                          labels=["Low","Medium","High"]).astype("object")
        return data, "SpendingCategory", f"Created from {score_col}: Low (≤39), Medium (40–69), High (70+)"
    amount_col = None
    for key in ["purchaseamount", "totalspend", "spendingamount", "amountspent", "annualspend"]:
        if key in lower_map:
            amount_col = lower_map[key]
            break
    if amount_col:
        values = pd.to_numeric(data[amount_col], errors="coerce")
        ranks = values.rank(pct=True)
        data["SpendingCategory"] = pd.cut(ranks, bins=[-0.001,.33,.67,1.001],
                                          labels=["Low","Medium","High"]).astype("object")
        return data, "SpendingCategory", f"Created using lower, middle and upper thirds of {amount_col}"
    return data, None, "No spending target found"

def build_models(frame, target):
    model_data = frame.dropna(subset=[target]).copy()
    y = model_data[target].astype(str)
    X = model_data.drop(columns=[target])
    # Remove likely identifiers and columns with a single value
    X = X.drop(columns=[c for c in X.columns if c.lower().replace("_","") in
                        ["customerid","id","userid","rownumber"]], errors="ignore")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable input features were found.")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    prep = ColumnTransformer(transformers)
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    models = {
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(
            n_estimators=220, random_state=42, class_weight="balanced", min_samples_leaf=2
        ))])
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "pipeline": model, "prediction": pred,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0)
        }
    return X, y, X_test, y_test, results, numeric, categorical

with st.sidebar:
    st.markdown("## ✦ SpendWise AI")
    st.caption("CUSTOMER SPENDING LAB")
    page = st.radio("NAVIGATION", ["Overview", "Explore Data", "Model Studio", "Predict Customer", "Business Insights"])
    st.divider()
    st.caption("Upload your own customer CSV or start with the included sample dataset.")
    uploaded = st.file_uploader("Customer dataset (.csv)", type=["csv"])

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        raw = load_sample()
        source_name = "sample_customers.csv"
    data = clean_frame(raw)
    data, target, target_note = create_target(data)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

st.markdown(f'<div class="section">ACTIVE DATASET · {source_name.upper()}</div>', unsafe_allow_html=True)
a,b,c,d = st.columns(4)
for col, label, value in [(a,"ROWS",f"{len(data):,}"),(b,"COLUMNS",f"{data.shape[1]:,}"),
                          (c,"DUPLICATES REMOVED",f"{len(raw)-len(data):,}"),
                          (d,"MISSING CELLS",f"{int(data.isna().sum().sum()):,}")]:
    col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

if page == "Overview":
    st.markdown("## Your customer intelligence workspace")
    left, right = st.columns([1.3,1])
    with left:
        st.markdown("### Project workflow")
        st.write("A practical end-to-end classification workflow: prepare customer records, explore behaviour, train two models, compare performance and predict a new customer's spending segment.")
        st.markdown("- **01 / Prepare** — standardize columns, remove duplicate rows and handle missing values in the model pipeline.")
        st.markdown("- **02 / Explore** — inspect distributions, relationships and spending patterns.")
        st.markdown("- **03 / Learn** — train Logistic Regression and Random Forest classifiers.")
        st.markdown("- **04 / Apply** — estimate Low, Medium or High spending category for a new customer.")
    with right:
        st.markdown("### Target readiness")
        if target:
            st.success(f"Target available: **{target}**")
            st.caption(target_note)
            st.write("The target is used to teach the models which customer spending group each record belongs to.")
        else:
            st.warning("A spending target could not be identified automatically.")
            st.write("Include a `SpendingCategory`, `SpendingScore`, or purchase amount column in your CSV.")
    st.markdown("### Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.download_button("⬇ Download cleaned dataset", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_customer_dataset.csv", mime="text/csv")

elif page == "Explore Data":
    st.markdown("## Explore customer behaviour")
    st.caption("Five views to understand customer composition, distributions and relationships.")
    st.dataframe(data.head(12), use_container_width=True)
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    st.markdown("### Data quality & summary")
    q1,q2 = st.columns(2)
    with q1:
        st.dataframe(pd.DataFrame({"Data type":data.dtypes.astype(str),
                                   "Missing":data.isna().sum(),
                                   "Unique":data.nunique(dropna=True)}), use_container_width=True)
    with q2:
        if numeric_cols:
            st.dataframe(data[numeric_cols].describe().T.round(2), use_container_width=True)
        else:
            st.info("No numeric columns found.")
    if numeric_cols:
        st.markdown("### 01 · Numeric feature distributions")
        chosen = st.selectbox("Choose a numeric feature", numeric_cols)
        fig, ax = plt.subplots(figsize=(8,3.2))
        sns.histplot(data[chosen].dropna(), kde=True, ax=ax, color="#20bfa2")
        ax.set_title(f"Distribution of {chosen}")
        st.pyplot(fig); plt.close(fig)
        st.markdown("### 02 · Feature correlation heatmap")
        corr = data[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(corr, annot=True, cmap="crest", fmt=".2f", ax=ax)
        st.pyplot(fig); plt.close(fig)
        st.markdown("### 03 · Numeric feature boxplot")
        box_col = st.selectbox("Select feature for boxplot", numeric_cols, key="box")
        fig, ax = plt.subplots(figsize=(8,2.8))
        sns.boxplot(x=data[box_col], ax=ax, color="#6d9ee8")
        st.pyplot(fig); plt.close(fig)
    if cat_cols:
        st.markdown("### 04 · Customer category composition")
        cat = st.selectbox("Choose a category", cat_cols)
        counts = data[cat].fillna("Missing").astype(str).value_counts().head(12)
        fig, ax = plt.subplots(figsize=(8,3.2))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#20bfa2")
        ax.set_xlabel("Number of customers"); ax.set_ylabel(cat)
        st.pyplot(fig); plt.close(fig)
    if target and numeric_cols:
        st.markdown("### 05 · Numeric features by spending segment")
        feat = st.selectbox("Choose a feature to compare with spending segment", [x for x in numeric_cols if x != target] or numeric_cols)
        fig, ax = plt.subplots(figsize=(8,3.2))
        sns.boxplot(data=data, x=target, y=feat, order=["Low","Medium","High"], ax=ax, palette="crest")
        ax.set_title(f"{feat} across spending categories")
        st.pyplot(fig); plt.close(fig)

elif page == "Model Studio":
    st.markdown("## Train & compare classification models")
    if not target:
        st.error("No target column found. Add SpendingCategory, SpendingScore, or a recognized purchase amount column.")
        st.stop()
    if st.button("⚡ Train both models", type="primary"):
        with st.spinner("Preparing features and training models..."):
            try:
                X, y, X_test, y_test, results, numeric, categorical = build_models(data, target)
                st.session_state["model_results"] = results
                st.session_state["model_y_test"] = y_test
                st.session_state["model_X_test"] = X_test
                st.session_state["model_features"] = X.columns.tolist()
                st.session_state["model_target"] = target
                st.session_state["model_numeric"] = numeric
                st.session_state["model_categorical"] = categorical
                st.success("Both models trained successfully.")
            except Exception as e:
                st.error(f"Training could not be completed: {e}")
    if "model_results" in st.session_state and st.session_state.get("model_target") == target:
        results = st.session_state["model_results"]
        metrics = pd.DataFrame([{"Model":name,"Accuracy":r["accuracy"],"Precision":r["precision"],
                                 "Recall":r["recall"],"F1-score":r["f1"]} for name,r in results.items()])
        st.markdown("### Model performance comparison")
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score"]}),
                     use_container_width=True)
        fig, ax = plt.subplots(figsize=(9,3.6))
        plot_data = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
        sns.barplot(data=plot_data, x="Metric", y="Score", hue="Model", ax=ax, palette=["#20bfa2","#6d9ee8"])
        ax.set_ylim(0,1.05); ax.set_ylabel("Weighted score")
        st.pyplot(fig); plt.close(fig)
        selected = st.selectbox("Inspect confusion matrix for", list(results.keys()))
        labels = sorted(st.session_state["model_y_test"].unique())
        cm = confusion_matrix(st.session_state["model_y_test"], results[selected]["prediction"], labels=labels)
        fig, ax = plt.subplots(figsize=(5,3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{selected} · Confusion matrix")
        st.pyplot(fig); plt.close(fig)
        rf = results["Random Forest"]["pipeline"]
        try:
            prep = rf.named_steps["prep"]
            names = prep.get_feature_names_out()
            importance = rf.named_steps["model"].feature_importances_
            imp = pd.DataFrame({"Feature":names,"Importance":importance}).sort_values("Importance",ascending=False).head(12)
            st.markdown("### Feature importance · Random Forest")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(data=imp, x="Importance", y="Feature", ax=ax, color="#20bfa2")
            st.pyplot(fig); plt.close(fig)
            st.dataframe(imp, use_container_width=True)
        except Exception:
            st.info("Feature importance is not available for this dataset configuration.")
    else:
        st.info("Click **Train both models** to generate the model comparison, confusion matrix and feature importance.")

elif page == "Predict Customer":
    st.markdown("## Predict a customer's spending segment")
    if not target:
        st.error("A target column is required before predictions can be made.")
        st.stop()
    if "model_results" not in st.session_state or st.session_state.get("model_target") != target:
        st.info("First open Model Studio and train both models.")
        st.stop()
    model_name = st.selectbox("Choose prediction model", list(st.session_state["model_results"].keys()))
    model = st.session_state["model_results"][model_name]["pipeline"]
    features = st.session_state["model_features"]
    with st.form("customer_prediction"):
        st.write("Enter customer information below.")
        values = {}
        for feature in features:
            if feature in st.session_state["model_numeric"]:
                series = pd.to_numeric(data[feature], errors="coerce").dropna()
                default = float(series.median()) if len(series) else 0.0
                values[feature] = st.number_input(feature, value=default)
            else:
                opts = data[feature].dropna().astype(str).value_counts().index.tolist()
                values[feature] = st.selectbox(feature, opts if opts else ["Unknown"])
        submitted = st.form_submit_button("Predict spending category")
    if submitted:
        sample = pd.DataFrame([values], columns=features)
        prediction = model.predict(sample)[0]
        st.markdown(f'<div class="hero"><div class="eyebrow">MODEL PREDICTION</div><div class="hero-title">{prediction.upper()} SPENDER</div><div class="hero-sub">Predicted using {model_name}.</div></div>', unsafe_allow_html=True)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(sample)[0]
            classes = model.classes_
            st.write("Prediction confidence by category")
            st.dataframe(pd.DataFrame({"Category":classes,"Probability":probs}).sort_values("Probability",ascending=False)
                         .style.format({"Probability":"{:.1%}"}), use_container_width=True)

elif page == "Business Insights":
    st.markdown("## Business insights")
    if not target:
        st.warning("Add a spending category or spending score column to generate segment insights.")
    else:
        segment_counts = data[target].value_counts(dropna=False)
        st.markdown("### Spending segment distribution")
        st.dataframe(segment_counts.rename_axis("Segment").reset_index(name="Customers"), use_container_width=True)
        if len(segment_counts):
            largest = str(segment_counts.index[0])
            share = segment_counts.iloc[0] / max(segment_counts.sum(),1)
            st.success(f"**Largest observed segment:** {largest} ({share:.1%} of records with a target).")
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        useful = [c for c in numeric_cols if c.lower().replace("_","") not in ["customerid","id"]]
        if useful:
            chosen = st.selectbox("Compare customer characteristics", useful)
            insight = data.groupby(target, dropna=True)[chosen].agg(["mean","median","count"]).round(2)
            st.markdown(f"### {chosen} by spending segment")
            st.dataframe(insight, use_container_width=True)
            if len(insight) >= 2:
                high = insight["mean"].idxmax()
                st.write(f"**Observation:** The **{high}** segment has the highest average {chosen} in this dataset.")
        st.markdown("### Suggested actions")
        st.write("- Tailor loyalty rewards to the needs and purchase patterns of each spending segment.")
        st.write("- Use high-spending customers to understand which products and experiences drive value.")
        st.write("- Test targeted offers for low- and medium-spending groups, then measure repeat purchases.")
        st.caption("These are descriptive insights from the uploaded dataset, not causal conclusions. Validate business decisions with additional data.")
st.markdown("---")
st.caption("SPENDWISE AI · Customer Spending Prediction · Built with Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn and Streamlit")
