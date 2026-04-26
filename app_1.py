import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CTR Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0f1117;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a2036 0%, #0f1117 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(64,120,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #5b8eff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7a8299;
    margin: 0;
    font-size: 0.95rem;
}

/* Cards */
.metric-card {
    background: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #5b8eff; }
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #5b8eff;
}
.metric-label {
    color: #7a8299;
    font-size: 0.82rem;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Prediction result */
.pred-box {
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
}
.pred-click    { background: #0d2a1a; border: 2px solid #22c55e; }
.pred-noclick  { background: #2a1010; border: 2px solid #ef4444; }
.pred-title    { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
.pred-prob     { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 600; }
.pred-subtitle { color: #9aa0b0; font-size: 0.9rem; margin-top: 6px; }

/* Table */
.comparison-table { width: 100%; border-collapse: collapse; }
.comparison-table th {
    background: #1a2036;
    color: #7a8299;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid #2a2f3e;
}
.comparison-table td {
    padding: 14px 16px;
    border-bottom: 1px solid #1e2335;
    color: #c8cfe0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
}
.comparison-table tr:hover td { background: #161b27; }
.best-row td { color: #5b8eff !important; }
.badge-best {
    background: #1a2c5c;
    color: #5b8eff;
    border: 1px solid #3a5aaa;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-runner {
    background: #1a261a;
    color: #4ade80;
    border: 1px solid #2a5a2a;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
}

/* Stacked bar */
.bar-track { background: #1e2335; border-radius: 4px; height: 8px; margin: 6px 0; }
.bar-fill  { height: 8px; border-radius: 4px; background: #5b8eff; }

/* Section title */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    color: #5b8eff;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2f3e;
}

/* Upload area */
.upload-hint {
    background: #161b27;
    border: 1px dashed #2a3a5c;
    border-radius: 10px;
    padding: 20px;
    color: #7a8299;
    font-size: 0.88rem;
    margin-bottom: 16px;
}

/* Info box */
.info-box {
    background: #0d1a2e;
    border-left: 3px solid #5b8eff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    color: #9ab0d0;
    font-size: 0.88rem;
    margin: 12px 0;
}

/* Justify box */
.justify-box {
    background: #0d1a10;
    border: 1px solid #2a4a2a;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 16px;
}
.justify-box h4 { color: #4ade80; margin: 0 0 8px 0; font-size: 1rem; }
.justify-box p  { color: #8aaa90; font-size: 0.88rem; margin: 0; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ─── Fake Model Results (replace with real loaded models) ─────────────────────
# In production: load actual PySpark/sklearn models from disk
MODEL_RESULTS = {
    "Logistic Regression": {
        "AUC-ROC": 0.7231, "Accuracy": 0.9052, "Precision": 0.8821,
        "Recall": 0.9052, "F1-Score": 0.8935, "Train Time": "2.4 min"
    },
    "Random Forest": {
        "AUC-ROC": 0.7618, "Accuracy": 0.9183, "Precision": 0.9074,
        "Recall": 0.9183, "F1-Score": 0.9128, "Train Time": "8.1 min"
    },
    "GBT": {
        "AUC-ROC": 0.7894, "Accuracy": 0.9347, "Precision": 0.9241,
        "Recall": 0.9347, "F1-Score": 0.9294, "Train Time": "14.3 min"
    }
}

BEST_MODEL = "GBT"

FEATURE_IMPORTANCE = {
    "ad_historical_ctr": 0.182,
    "user_ctr": 0.154,
    "price": 0.121,
    "hour": 0.098,
    "age_level": 0.087,
    "shopping_level": 0.079,
    "pvalue_level": 0.071,
    "cate_id": 0.063,
    "day_of_week": 0.058,
    "time_segment": 0.049,
    "final_gender_code": 0.038,
}

def predict_ctr(features: dict, model_name: str) -> float:
    """Simulate CTR prediction — replace with real model.predict()"""
    np.random.seed(hash(str(features) + model_name) % (2**31))
    base = 0.051  # dataset CTR
    # simple heuristic for demo
    boost = 0
    if features.get("shopping_level") == 3: boost += 0.04
    if features.get("pvalue_level") == 3:    boost += 0.03
    if features.get("hour") in range(19, 23): boost += 0.02
    if features.get("age_level") in [3, 4]:  boost += 0.02
    model_boost = {"Logistic Regression": 0, "Random Forest": 0.005, "GBT": 0.008}
    prob = min(max(base + boost + model_boost[model_name] + np.random.normal(0, 0.005), 0.01), 0.99)
    return round(float(prob), 4)

def batch_predict(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Run predictions on uploaded CSV — replace with real pipeline"""
    results = df.copy()
    probs = []
    for _, row in df.iterrows():
        np.random.seed(int(hash(str(row.values.tolist())) % (2**31)))
        probs.append(round(float(np.random.beta(1.5, 27)), 4))
    results["click_probability"] = probs
    results["prediction"] = (results["click_probability"] > 0.05).astype(int)
    results["label"] = results["prediction"].map({1: "CLICK", 0: "NO CLICK"})
    return results

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;
                    color:#5b8eff;font-weight:600;'>CTR Predict</div>
        <div style='color:#4a5068;font-size:0.78rem;margin-top:2px;'>Taobao Ad Click System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigation",
        ["Single Prediction", "Batch Upload", "Model Comparison"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style='color:#4a5068;font-size:0.78rem;line-height:1.7;'>
        <b style='color:#7a8299;'>Dataset</b><br>
        Taobao / Alimama CTR<br>
        26.5M records · 3 tables<br><br>
        <b style='color:#7a8299;'>Models</b><br>
        Logistic Regression<br>
        Random Forest (100 trees)<br>
        GBT (maxIter=20)<br><br>
       <br>
     
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "Single Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1> Single Prediction</h1>
        <p>Enter user & ad features to predict click probability</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-title'>User Profile</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            age_level = st.selectbox("Age Level", [1,2,3,4,5,6,7],
                format_func=lambda x: {1:"<18",2:"18-24",3:"25-29",4:"30-34",5:"35-39",6:"40-49",7:"50+"}[x])
            occupation = st.selectbox("Student?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            pvalue_level = st.selectbox("Consumption Grade", [1,2,3],
                format_func=lambda x: {1:"Low",2:"Mid",3:"High"}[x])
            shopping_level = st.selectbox("Shopping Depth", [1,2,3],
                format_func=lambda x: {1:"Shallow",2:"Moderate",3:"Deep"}[x])
            city_level = st.selectbox("City Level", [1,2,3,4])

        st.markdown("<div class='section-title' style='margin-top:20px;'>Ad Features</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            cate_id = st.number_input("Category ID", min_value=1, max_value=13000, value=6406)
            price = st.number_input("Price (¥)", min_value=0.0, max_value=10000.0, value=199.0, step=10.0)
        with c4:
            brand_id = st.number_input("Brand ID", min_value=0, max_value=500000, value=95471)
            hour = st.slider("Hour of Day", 0, 23, 14)

        st.markdown("<div class='section-title' style='margin-top:20px;'>Select Model</div>", unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Model", list(MODEL_RESULTS.keys()),
            index=2,
            label_visibility="collapsed"
        )

        predict_btn = st.button(" Predict Click Probability", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            features = {
                "final_gender_code": gender, "age_level": age_level,
                "occupation": occupation, "pvalue_level": pvalue_level,
                "shopping_level": shopping_level, "new_user_class_level": city_level,
                "cate_id": cate_id, "price": price, "brand": brand_id, "hour": hour
            }
            with st.spinner("Predicting..."):
                time.sleep(0.4)
                prob = predict_ctr(features, selected_model)

            click = prob >= 0.05
            box_class = "pred-click" if click else "pred-noclick"
            label = " WILL CLICK" if click else "❌ WON'T CLICK"
            color = "#22c55e" if click else "#ef4444"

            st.markdown(f"""
            <div class='pred-box {box_class}'>
                <div class='pred-title' style='color:{color};'>{label}</div>
                <div class='pred-prob' style='color:{color};'>{prob*100:.1f}%</div>
                <div class='pred-subtitle'>Click Probability · {selected_model}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Feature Contribution (Top 5)</div>", unsafe_allow_html=True)

            top_feats = list(FEATURE_IMPORTANCE.items())[:5]
            for fname, imp in top_feats:
                pct = int(imp * 100)
                st.markdown(f"""
                <div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;
                                color:#9aa0b0;font-size:0.82rem;margin-bottom:4px;'>
                        <span>{fname}</span><span>{pct}%</span>
                    </div>
                    <div class='bar-track'>
                        <div class='bar-fill' style='width:{pct*3}px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='info-box'>
                Model used: <b>{selected_model}</b> · 
                AUC-ROC: <b>{MODEL_RESULTS[selected_model]['AUC-ROC']}</b> · 
                F1: <b>{MODEL_RESULTS[selected_model]['F1-Score']}</b>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;color:#4a5068;'>
                <div style='font-size:3rem;margin-bottom:12px;'></div>
                <div style='font-size:1rem;color:#5a6080;'>Fill in the form and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Upload (BONUS)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Upload":
    st.markdown("""
    <div class='main-header'>
        <h1> Batch Upload </h1>
        <p>Upload a CSV file and get predictions for all rows at once</p>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_cfg = st.columns([2, 1], gap="large")

    with col_cfg:
        st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)
        batch_model = st.selectbox("Model for Batch", list(MODEL_RESULTS.keys()), index=2)
        threshold = st.slider("Click Threshold", 0.01, 0.20, 0.05, 0.005,
                              help="Probability above this = predicted click")
        show_only_clicks = st.checkbox("Show only predicted clicks", value=False)

        st.markdown(f"""
        <div class='info-box' style='margin-top:16px;'>
            <b>Selected:</b> {batch_model}<br>
            <b>AUC-ROC:</b> {MODEL_RESULTS[batch_model]['AUC-ROC']}<br>
            <b>Threshold:</b> {threshold}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:20px;'>Expected CSV Columns</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;
                    background:#0f1117;border:1px solid #2a2f3e;border-radius:8px;
                    padding:14px;color:#7a8299;line-height:2;'>
            user, adgroup_id, cate_id,<br>
            brand, price, pid,<br>
            final_gender_code, age_level,<br>
            pvalue_level, shopping_level,<br>
            occupation
        </div>
        """, unsafe_allow_html=True)

    with col_up:
        st.markdown("<div class='section-title'>Upload Dataset</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='upload-hint'>
             Upload a CSV file with the required columns.
            The model will predict click probability for each row.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            with st.spinner("Loading file..."):
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.success(f" Loaded {len(df_uploaded):,} rows · {len(df_uploaded.columns)} columns")
                    st.dataframe(df_uploaded.head(5), use_container_width=True,
                                 hide_index=True, height=180)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    df_uploaded = None

            if df_uploaded is not None and st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner(f"Running {batch_model} on {len(df_uploaded):,} rows..."):
                    time.sleep(0.8)
                    results_df = batch_predict(df_uploaded, batch_model)
                    results_df["prediction"] = (results_df["click_probability"] >= threshold).astype(int)
                    results_df["label"] = results_df["prediction"].map({1: "CLICK", 0: "NO CLICK"})

                st.markdown("<div class='section-title' style='margin-top:20px;'>Results Summary</div>",
                            unsafe_allow_html=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                total      = len(results_df)
                n_clicks   = results_df["prediction"].sum()
                n_noclicks = total - n_clicks
                ctr        = n_clicks / total * 100

                for col, val, lbl in zip(
                    [mc1, mc2, mc3, mc4],
                    [total, n_clicks, n_noclicks, f"{ctr:.1f}%"],
                    ["Total Rows", "Predicted Clicks", "No Clicks", "Predicted CTR"]
                ):
                    with col:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-value'>{val:,}</div>
                            <div class='metric-label'>{lbl}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                display_df = results_df[results_df["prediction"] == 1] if show_only_clicks else results_df
                st.dataframe(
                    display_df.style.applymap(
                        lambda v: "color:#22c55e;" if v == "CLICK" else "color:#ef4444;",
                        subset=["label"]
                    ).format({"click_probability": "{:.4f}"}),
                    use_container_width=True, hide_index=True, height=320
                )

                # Download
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=" Download Results CSV",
                    data=csv_bytes,
                    file_name=f"ctr_predictions_{batch_model.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            # Demo with sample data
            st.markdown("<div class='section-title'>Or Try with Sample Data</div>", unsafe_allow_html=True)
            if st.button(" Run Demo on Sample (100 rows)", use_container_width=True):
                np.random.seed(42)
                sample_df = pd.DataFrame({
                    "user":               np.random.randint(100000, 999999, 100),
                    "adgroup_id":         np.random.randint(1, 846811, 100),
                    "cate_id":            np.random.choice([6406, 392, 7211, 5003, 2345], 100),
                    "brand":              np.random.randint(1, 500000, 100),
                    "price":              np.random.uniform(9.9, 999.0, 100).round(2),
                    "final_gender_code":  np.random.choice([1, 2], 100),
                    "age_level":          np.random.randint(1, 8, 100),
                    "pvalue_level":       np.random.choice([1, 2, 3], 100),
                    "shopping_level":     np.random.choice([1, 2, 3], 100),
                    "occupation":         np.random.choice([0, 1], 100),
                })
                with st.spinner("Running predictions..."):
                    time.sleep(0.6)
                    results_df = batch_predict(sample_df, batch_model)
                    results_df["prediction"] = (results_df["click_probability"] >= threshold).astype(int)
                    results_df["label"] = results_df["prediction"].map({1: "CLICK", 0: "NO CLICK"})

                n_clicks = results_df["prediction"].sum()
                st.success(f" Done! {n_clicks} clicks predicted out of 100 rows")
                st.dataframe(
                    results_df.style.format({"click_probability": "{:.4f}"}),
                    use_container_width=True, hide_index=True, height=300
                )
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(" Download Demo Results", data=csv_bytes,
                                   file_name="demo_predictions.csv", mime="text/csv",
                                   use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Comparison (BONUS)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown("""
    <div class='main-header'>
        <h1> Model Comparison </h1>
        <p>Compare all 3 models · Select the best · Understand the results</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Table ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)

    table_df = pd.DataFrame([
        {"Model": "Logistic Regression", "AUC-ROC": 0.7231, "Accuracy": 0.9052,
         "Precision": 0.8821, "Recall": 0.9052, "F1-Score": 0.8935,
         "Train Time": "2.4 min", "Rank": "3rd"},
        {"Model": "Random Forest", "AUC-ROC": 0.7618, "Accuracy": 0.9183,
         "Precision": 0.9074, "Recall": 0.9183, "F1-Score": 0.9128,
         "Train Time": "8.1 min", "Rank": "2nd"},
        {"Model": "GBT the BEST", "AUC-ROC": 0.7894, "Accuracy": 0.9347,
         "Precision": 0.9241, "Recall": 0.9347, "F1-Score": 0.9294,
         "Train Time": "14.3 min", "Rank": "1st"},
    ])
    st.dataframe(
        table_df.style.highlight_max(
            subset=["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"],
            color="#1a3a2a"
        ).format({
            "AUC-ROC": "{:.4f}", "Accuracy": "{:.4f}",
            "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True,
        height=150
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_bar, col_feat = st.columns([1, 1], gap="large")

    with col_bar:
        st.markdown("<div class='section-title'>Metrics Comparison</div>", unsafe_allow_html=True)
        models  = list(MODEL_RESULTS.keys())
        metrics = ["AUC-ROC", "Accuracy", "F1-Score"]
        x       = np.arange(len(metrics))
        width   = 0.25
        colors  = ["#4a5580", "#4ade80", "#5b8eff"]

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#161b27")
        ax.set_facecolor("#161b27")

        for i, (model, color) in enumerate(zip(models, colors)):
            vals = [MODEL_RESULTS[model][m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, color="#9aa0b0")

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, color="#9aa0b0", fontsize=9)
        ax.set_ylim(0.68, 0.98)
        ax.tick_params(colors="#4a5068", labelsize=8)
        ax.spines[:].set_color("#2a2f3e")
        ax.yaxis.label.set_color("#7a8299")
        ax.set_ylabel("Score", color="#7a8299", fontsize=9)
        ax.legend(fontsize=8, labelcolor="#9aa0b0",
                  facecolor="#1a2036", edgecolor="#2a2f3e")
        ax.set_title("Model Performance", color="#c8cfe0", fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_feat:
        st.markdown("<div class='section-title'>Feature Importance (GBT)</div>", unsafe_allow_html=True)
        feat_df = pd.DataFrame(list(FEATURE_IMPORTANCE.items()),
                               columns=["Feature", "Importance"]).sort_values("Importance")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#161b27")
        ax2.set_facecolor("#161b27")

        colors_fi = ["#5b8eff" if i >= len(feat_df) - 3 else "#2a3a5c"
                     for i in range(len(feat_df))]
        bars = ax2.barh(feat_df["Feature"], feat_df["Importance"],
                        color=colors_fi, height=0.65)
        for bar, val in zip(bars, feat_df["Importance"]):
            ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=7.5, color="#9aa0b0")

        ax2.tick_params(colors="#7a8299", labelsize=8)
        ax2.spines[:].set_color("#2a2f3e")
        ax2.set_xlabel("Importance", color="#7a8299", fontsize=9)
        ax2.set_title("GBT Feature Importance", color="#c8cfe0", fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>ROC Curves</div>", unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor("#161b27")
    ax3.set_facecolor("#161b27")
    roc_colors = {"Logistic Regression": "#4a5580", "Random Forest": "#4ade80", "GBT": "#5b8eff"}
    np.random.seed(42)

    for model, color in roc_colors.items():
        auc_val = MODEL_RESULTS[model]["AUC-ROC"]
        t   = np.linspace(0, 1, 100)
        fpr = t ** (1 / (auc_val * 2))
        tpr = 1 - (1 - t) ** (auc_val * 2)
        fpr = np.clip(fpr, 0, 1); tpr = np.clip(tpr, 0, 1)
        ax3.plot(fpr, tpr, color=color, linewidth=2.2,
                 label=f"{model} (AUC={auc_val})")

    ax3.plot([0, 1], [0, 1], color="#2a2f3e", linestyle="--", linewidth=1.2)
    ax3.set_xlabel("False Positive Rate", color="#7a8299", fontsize=9)
    ax3.set_ylabel("True Positive Rate", color="#7a8299", fontsize=9)
    ax3.set_title("ROC Curves — All Models", color="#c8cfe0", fontsize=11, pad=10)
    ax3.tick_params(colors="#4a5068", labelsize=8)
    ax3.spines[:].set_color("#2a2f3e")
    ax3.legend(fontsize=8.5, labelcolor="#9aa0b0", facecolor="#1a2036", edgecolor="#2a2f3e")
    plt.tight_layout()
    col_roc, col_just = st.columns([1, 1], gap="large")
    with col_roc:
        st.pyplot(fig3, use_container_width=True)
    plt.close()

    # ── Justification ─────────────────────────────────────────────────────────
    with col_just:
       
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Quick Model Selector</div>", unsafe_allow_html=True)
        chosen = st.selectbox("Select a model to inspect", list(MODEL_RESULTS.keys()),
                              index=2, label_visibility="collapsed")
        res = MODEL_RESULTS[chosen]
        for metric, val in res.items():
            if metric != "Train Time":
                bar_w = int(float(val) * 180)
                st.markdown(f"""
                <div style='margin-bottom:8px;'>
                    <div style='display:flex;justify-content:space-between;
                                color:#9aa0b0;font-size:0.8rem;margin-bottom:3px;'>
                        <span>{metric}</span><span>{val}</span>
                    </div>
                    <div class='bar-track'>
                        <div class='bar-fill' style='width:{bar_w}px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='color:#4a5068;font-size:0.8rem;margin-top:8px;'>
            ⏱ Training time: {res['Train Time']}
        </div>
        """, unsafe_allow_html=True)
