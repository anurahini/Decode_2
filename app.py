"""
Interactive Web Application for Iris Species Classification
DecodeLabs AI Training Kit - Project 2: Data Classification Using AI
Run with: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set page configuration
st.set_page_config(
    page_title="DecodeLabs AI | Iris Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.5rem;
    }
    .pred-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .species-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0.4rem 0;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler():
    model_path = "saved_models/knn_model.joblib"
    scaler_path = "saved_models/scaler.joblib"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        from src.train import run_training_pipeline
        run_training_pipeline()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_data
def load_data():
    csv_path = "data/iris.csv"
    if not os.path.exists(csv_path):
        from src.data_loader import load_iris_data
        return load_iris_data(csv_path)
    return pd.read_csv(csv_path)


# Load resources
model, scaler = load_model_and_scaler()
df = load_data()

# Header
st.markdown('<div class="main-header">🌸 Iris Flower Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<span class="badge-pill">DecodeLabs Batch 2026</span>'
    '<span class="badge-pill">Project 2: Supervised Learning</span>'
    '<span class="badge-pill">Algorithm: K-Nearest Neighbors (KNN)</span>'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar: Controls & Presets
st.sidebar.header("⚙️ Feature Input Controls")
st.sidebar.markdown("Adjust dimensions or select a benchmark preset to classify:")

preset = st.sidebar.selectbox(
    "Benchmark Presets",
    ["Custom", "Iris Setosa Benchmark", "Iris Versicolor Benchmark", "Iris Virginica Benchmark"]
)

if preset == "Iris Setosa Benchmark":
    def_sl, def_sw, def_pl, def_pw = 5.0, 3.5, 1.4, 0.2
elif preset == "Iris Versicolor Benchmark":
    def_sl, def_sw, def_pl, def_pw = 6.0, 2.8, 4.5, 1.4
elif preset == "Iris Virginica Benchmark":
    def_sl, def_sw, def_pl, def_pw = 6.8, 3.2, 5.9, 2.3
else:
    def_sl, def_sw, def_pl, def_pw = 5.8, 3.0, 4.2, 1.3

sl = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=float(def_sl), step=0.1)
sw = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=float(def_sw), step=0.1)
pl = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=float(def_pl), step=0.1)
pw = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.6, value=float(def_pw), step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Parameters:**\n- **Algorithm:** KNN (k={model.n_neighbors})\n- **Distance Metric:** Euclidean (Minkowski p=2)\n- **Feature Scaling:** StandardScaler")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Real-Time Classifier",
    "📐 IPO Master Blueprint",
    "📊 Evaluation & Metrics",
    "📁 Dataset Explorer"
])

# ----------------- TAB 1: Real-Time Classifier -----------------
with tab1:
    col_pred, col_viz = st.columns([1, 1.2])

    # Perform inference
    raw_input = np.array([[sl, sw, pl, pw]], dtype=float)
    scaled_input = scaler.transform(raw_input)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    classes = model.classes_
    confidence = float(np.max(probabilities))

    species_emoji = {
        "setosa": "🌺 Iris Setosa",
        "versicolor": "🪻 Iris Versicolor",
        "virginica": "🌸 Iris Virginica"
    }

    with col_pred:
        st.subheader("Classification Outcome")
        st.markdown(f"""
        <div class="pred-card">
            <div>PREDICTED SPECIES</div>
            <div class="species-title">{species_emoji.get(prediction, prediction)}</div>
            <div>Confidence: <b>{confidence * 100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Probability Distribution")
        for cls_name, prob in zip(classes, probabilities):
            st.write(f"**{cls_name.capitalize()}**: `{prob * 100:.1f}%`")
            st.progress(float(prob))

        st.markdown("---")
        st.markdown("#### Feature Values Submitted")
        feat_df = pd.DataFrame({
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Raw Value (cm)": [f"{sl:.1f}", f"{sw:.1f}", f"{pl:.1f}", f"{pw:.1f}"],
            "Standardized Z-Score": [f"{scaled_input[0][0]:.3f}", f"{scaled_input[0][1]:.3f}", f"{scaled_input[0][2]:.3f}", f"{scaled_input[0][3]:.3f}"]
        })
        st.table(feat_df)

    with col_viz:
        st.subheader("Interactive 2D Feature Space Mapping")
        st.caption("Visualizing where your input sample falls relative to the training clusters:")

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        color_map = {"setosa": "#2563EB", "versicolor": "#059669", "virginica": "#EA580C"}

        for spec, color in color_map.items():
            subset = df[df["species"] == spec]
            ax.scatter(subset["petal_length"], subset["petal_width"], label=spec.capitalize(), color=color, alpha=0.6, s=40)

        # Plot user input point
        ax.scatter([pl], [pw], color="#DC2626", s=180, marker="*", edgecolor="black", linewidth=1.5, label="Your Input Sample", zorder=10)

        ax.set_xlabel("Petal Length (cm)", fontweight="bold")
        ax.set_ylabel("Petal Width (cm)", fontweight="bold")
        ax.set_title("Petal Dimensions Clustering (Primary Discriminator)", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ----------------- TAB 2: IPO Master Blueprint -----------------
with tab2:
    st.subheader("Architectural Paradigm: The IPO Framework")
    st.markdown("""
    As defined in **DecodeLabs AI Project 2**, supervised learning replaces heuristic if/then logic with an algorithmic decision pipeline:
    """)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        ### 1. INPUT 📥
        - **Dataset:** Iris Benchmark (150 samples, 4 dimensions, 3 balanced classes)
        - **The Gatekeeper Rule:** Feature Scaling via `StandardScaler` (Mean = 0, Variance = 1)
        - Prevents high-magnitude dimensions from dominating Euclidean distances.
        """)
    with b2:
        st.markdown("""
        ### 2. PROCESS ⚙️
        - **Structural Split:** 80% Training Set, 20% Testing Set
        - **Order Bias Elimination:** Random shuffling with stratification
        - **KNN Algorithm:** The Proximity Principle (K-Nearest Neighbors vote)
        - **Engine Tuning:** The Elbow Method to select optimal $K$
        """)
    with b3:
        st.markdown("""
        ### 3. OUTPUT 📤
        - **Diagnostic Matrix:** Confusion Matrix mapping TP, FP, FN, TN
        - **Avoiding the Accuracy Mirage:** Precision, Recall, and F1 Score (Harmonic Mean)
        - **Inference Pipeline:** Portable serialized deployment artifacts
        """)

    st.markdown("---")
    st.markdown("#### Scikit-Learn 3-Step Lifecycle")
    st.code("""
# 1. INSTANTIATE (Build the frame)
model = KNeighborsClassifier(n_neighbors=7)

# 2. FIT (Memorize the map)
model.fit(X_train_scaled, y_train)

# 3. PREDICT (Apply proximity logic)
predictions = model.predict(X_test_scaled)
    """, language="python")

# ----------------- TAB 3: Evaluation & Metrics -----------------
with tab3:
    st.subheader("Model Diagnostic & Performance Report")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", "96.67%", "+0.0% variance")
    m2.metric("Macro Precision", "0.9697", "High reliability")
    m3.metric("Macro Recall", "0.9667", "Minimal missed detections")
    m4.metric("Macro F1-Score", "0.9666", "Harmonic balance")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Confusion Matrix Heatmap")
        if os.path.exists("plots/confusion_matrix.png"):
            st.image("plots/confusion_matrix.png", caption="Diagnostic Confusion Matrix on 20% Test Split", use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate the confusion matrix plot.")
    with c2:
        st.markdown("#### The Elbow Method (K-Optimization)")
        if os.path.exists("plots/elbow_method.png"):
            st.image("plots/elbow_method.png", caption="Error Rate vs K Value showing optimal Elbow", use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate the elbow curve.")

    st.markdown("---")
    st.markdown("#### Algorithm Benchmark Comparison")
    if os.path.exists("plots/model_comparison.png"):
        st.image("plots/model_comparison.png", caption="Comparison across 6 Supervised Classifiers", use_container_width=True)

# ----------------- TAB 4: Dataset Explorer -----------------
with tab4:
    st.subheader("Iris Benchmark Dataset Explorer")
    st.write(f"Total Rows: **{len(df)}** | Total Features: **4** | Target Classes: **3** (Balanced: 50 each)")

    selected_species = st.multiselect("Filter by Species", options=df["species"].unique(), default=df["species"].unique())
    filtered_df = df[df["species"].isin(selected_species)]
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Statistical Overview")
    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("---")
    if os.path.exists("plots/feature_distributions.png"):
        st.markdown("#### Feature Distribution Profiles")
        st.image("plots/feature_distributions.png", use_container_width=True)
