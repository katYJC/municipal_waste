import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Municipal Waste | Streamlit Demo",
    page_icon="🗑️",
    layout="wide"
)

st.title("🗑️ Municipal Waste (一般廢棄物)｜Streamlit Demo")
st.caption("EDA・Correlation・Modeling (Linear Regression vs Random Forest)")

# -------------------------
# Helpers
# -------------------------
REQUIRED_COLS = ["year", "county", "garbagegenerated", "garbageclearance", "garbagerecycled", "foodwastesrecycled"]

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out

def metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def corr_heatmap_matplotlib(corr: pd.DataFrame, title: str = "Correlation Heatmap"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic cleaning
    for c in ["garbagegenerated","garbageclearance","garbagerecycled","foodwastesrecycled"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["county"] = df["county"].astype(str)

    # Missing value handling (robust default)
    for c in ["garbagegenerated","garbageclearance","garbagerecycled","foodwastesrecycled"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    if df["year"].isna().any():
        df["year"] = df["year"].fillna(df["year"].median()).astype("Int64")

    # Consistency clip (logical constraints)
    df["garbagerecycled"] = df[["garbagerecycled","garbagegenerated"]].min(axis=1)
    df["foodwastesrecycled"] = df[["foodwastesrecycled","garbagegenerated"]].min(axis=1)
    df["garbageclearance"] = df[["garbageclearance","garbagegenerated"]].min(axis=1)

    # Feature engineering
    df["recycling_rate"] = safe_div(df["garbagerecycled"], df["garbagegenerated"]).clip(0, 1)
    df["foodwaste_ratio"] = safe_div(df["foodwastesrecycled"], df["garbagegenerated"]).clip(0, 1)
    return df

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    # Try utf-8-sig first, then fallback
    raw = uploaded_file.read()
    for enc in ["utf-8-sig", "utf-8", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    # Last resort
    return pd.read_csv(io.BytesIO(raw))

# -------------------------
# Sidebar: Data input
# -------------------------
st.sidebar.header("1) Data")
mode = st.sidebar.radio("Choose data source", ["Upload CSV", "Use local sample (path)"], index=0)

df_raw = None

if mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if up is not None:
        df_raw = load_csv(up)
else:
    path = st.sidebar.text_input("Local CSV path", value="一般廢棄物清理情況資料.csv")
    try:
        df_raw = pd.read_csv(path, encoding="utf-8-sig")
        st.sidebar.success("Loaded local file.")
    except Exception as e:
        st.sidebar.warning(f"Cannot load file: {e}")

if df_raw is None:
    st.info("👈 Please upload a CSV to start.")
    st.stop()

# Validate columns
missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}\n\nExpected: {REQUIRED_COLS}")
    st.stop()

df = ensure_features(df_raw)

# -------------------------
# Layout tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "📊 EDA", "🔗 Correlation", "🤖 Modeling"])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Year range", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c4.metric("Counties", f"{df['county'].nunique():,}")

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().to_frame("missing_count"), use_container_width=True)

    st.subheader("Basic Statistics")
    num_cols = ["garbagegenerated","garbageclearance","garbagerecycled","foodwastesrecycled","recycling_rate","foodwaste_ratio"]
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

# -------------------------
# Tab 2: EDA
# -------------------------
with tab2:
    st.subheader("Yearly Trend (Total Waste Generated)")
    yearly = df.groupby("year")["garbagegenerated"].sum().sort_index()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(yearly.index.astype(int), yearly.values, marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Garbage Generated")
    ax.set_title("Annual Trend of Waste Generation")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Top 10 Counties (Average Waste Generated)")
    top10 = (
        df.groupby("county")["garbagegenerated"]
        .mean()
        .sort_values(ascending=True)
        .tail(10)
    )

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.barh(top10.index, top10.values)
    ax.set_xlabel("Average Garbage Generated")
    ax.set_ylabel("County")
    ax.set_title("Top 10 Counties by Average Waste Generation")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Scatter: Recycling Rate vs Waste Generated")
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    ax.scatter(df["recycling_rate"], df["garbagegenerated"], alpha=0.6)
    ax.set_xlabel("Recycling Rate")
    ax.set_ylabel("Garbage Generated")
    ax.set_title("Recycling Rate vs Waste Generated")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# -------------------------
# Tab 3: Correlation
# -------------------------
with tab3:
    st.subheader("Correlation Matrix (Numeric Features)")
    corr_cols = ["garbagegenerated","garbageclearance","garbagerecycled","foodwastesrecycled","recycling_rate","foodwaste_ratio"]
    corr = df[corr_cols].corr()

    st.dataframe(corr, use_container_width=True)

    st.subheader("Correlation Heatmap")
    st.pyplot(corr_heatmap_matplotlib(corr), clear_figure=True)

# -------------------------
# Tab 4: Modeling
# -------------------------
with tab4:
    st.subheader("Model Setup")
    target = st.selectbox("Target (y)", ["garbagegenerated"], index=0)

    feature_options = ["garbageclearance","garbagerecycled","foodwastesrecycled","recycling_rate","foodwaste_ratio"]
    default_feats = ["garbageclearance","garbagerecycled","foodwastesrecycled","recycling_rate","foodwaste_ratio"]
    features = st.multiselect("Features (X)", feature_options, default=default_feats)

    colA, colB, colC = st.columns(3)
    test_size = colA.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    seed = colB.number_input("Random seed", value=42, step=1)
    n_estimators = colC.slider("RF n_estimators", 50, 600, 300, 50)

    if len(features) == 0:
        st.warning("Please select at least 1 feature.")
        st.stop()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed)
    )

    run = st.button("Train & Evaluate", type="primary")
    if run:
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        m_lr = metrics(y_test, pred_lr)

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        m_rf = metrics(y_test, pred_rf)

        # Compare
        comp = pd.DataFrame([
            {"Model":"Linear Regression", **m_lr},
            {"Model":"Random Forest", **m_rf},
        ])

        st.subheader("Model Performance Comparison")
        st.dataframe(comp, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Actual vs Predicted (Linear Regression)**")
            fig = plt.figure(figsize=(5.5, 5.5))
            ax = fig.add_subplot(111)
            ax.scatter(y_test, pred_lr, alpha=0.6)
            mn, mx = min(y_test.min(), pred_lr.min()), max(y_test.max(), pred_lr.max())
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("LR: Actual vs Predicted")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        with c2:
            st.markdown("**Actual vs Predicted (Random Forest)**")
            fig = plt.figure(figsize=(5.5, 5.5))
            ax = fig.add_subplot(111)
            ax.scatter(y_test, pred_rf, alpha=0.6)
            mn, mx = min(y_test.min(), pred_rf.min()), max(y_test.max(), pred_rf.max())
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("RF: Actual vs Predicted")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        st.subheader("Random Forest Feature Importance")
        fi = pd.DataFrame({
            "Feature": features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.barh(fi["Feature"], fi["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("RF Feature Importance")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Download predictions
        st.subheader("Download Predictions (Test Set)")
        out = X_test.copy()
        out["actual"] = y_test.values
        out["pred_lr"] = pred_lr
        out["pred_rf"] = pred_rf
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download CSV (test predictions)",
            data=csv_bytes,
            file_name="predictions_test.csv",
            mime="text/csv"
        )
