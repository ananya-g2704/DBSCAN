import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="DBSCAN Customer Segmentation",
    page_icon="🔮",
    layout="wide"
)

# ==================================================
# CUSTOM THEME
# ==================================================

st.markdown("""
<style>

.stApp{
    background-color:#F8F4FF;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#E9D5FF;
}

/* Headers */
h1,h2,h3,h4,h5,h6{
    color:#4A148C !important;
}

/* General Text */
p, span, label, div{
    color:#2D1B4E;
}

/* Metrics */
div[data-testid="metric-container"]{
    background-color:white;
    border:2px solid #C77DFF;
    padding:15px;
    border-radius:15px;
}

/* Buttons */
.stButton > button{
    background-color:#7B2CBF;
    color:white !important;
    border:none;
    border-radius:10px;
    padding:10px 20px;
}

.stButton > button:hover{
    background-color:#5A189A;
}

/* Select Box */
[data-baseweb="select"] *{
    color:black !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("data/CC GENERAL.csv")

# ==================================================
# HANDLE MISSING VALUES
# ==================================================

df.fillna(
    df.median(numeric_only=True),
    inplace=True
)

# ==================================================
# FEATURES
# ==================================================

X = df.drop(
    columns=["CUST_ID"]
)

# ==================================================
# SCALING
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ DBSCAN Settings")

eps = st.sidebar.slider(
    "EPS",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1
)

min_samples = st.sidebar.slider(
    "Min Samples",
    min_value=2,
    max_value=50,
    value=10
)

# ==================================================
# TRAIN MODEL
# ==================================================

model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = model.fit_predict(X_scaled)

# ==================================================
# CLUSTER INFO
# ==================================================

num_clusters = len(set(clusters))

if -1 in clusters:
    num_clusters -= 1

noise_points = np.sum(clusters == -1)

# ==================================================
# TITLE
# ==================================================

st.title("🔮 DBSCAN Customer Segmentation Dashboard")

st.markdown("""
This dashboard uses **DBSCAN (Density-Based Spatial Clustering)** to discover customer groups and detect outliers.
""")

# ==================================================
# OVERVIEW CARDS
# ==================================================

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with c2:
    st.metric(
        "Features",
        X.shape[1]
    )

with c3:
    st.metric(
        "Clusters Found",
        num_clusters
    )

with c4:
    st.metric(
        "Noise Points",
        noise_points
    )

# ==================================================
# DATA PREVIEW
# ==================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# ==================================================
# HEATMAP
# ==================================================

st.subheader("🔥 Correlation Heatmap")

fig1, ax1 = plt.subplots(
    figsize=(10,6)
)

sns.heatmap(
    X.corr(numeric_only=True),
    cmap="Purples",
    ax=ax1
)

st.pyplot(fig1)

# ==================================================
# PCA VISUALIZATION
# ==================================================

st.subheader("🎯 Cluster Visualization")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

fig2, ax2 = plt.subplots(
    figsize=(8,6)
)

scatter = ax2.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

ax2.set_title(
    "DBSCAN Clusters"
)

ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")

st.pyplot(fig2)

# ==================================================
# OUTLIERS
# ==================================================

st.subheader("🚨 Outlier Detection")

noise_mask = clusters == -1

fig3, ax3 = plt.subplots(
    figsize=(8,6)
)

ax3.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c="lightgray",
    alpha=0.5
)

if noise_points > 0:

    ax3.scatter(
        X_pca[noise_mask,0],
        X_pca[noise_mask,1],
        c="red",
        marker="x",
        s=100,
        label="Outliers"
    )

ax3.legend()

st.pyplot(fig3)

# ==================================================
# SILHOUETTE SCORE
# ==================================================

st.subheader("⭐ Silhouette Score")

try:

    valid_clusters = len(set(clusters))

    if valid_clusters > 1:

        score = silhouette_score(
            X_scaled,
            clusters
        )

        st.success(
            f"Silhouette Score : {score:.3f}"
        )

    else:

        st.warning(
            "Silhouette Score cannot be calculated with only one cluster."
        )

except:

    st.warning(
        "Silhouette Score not available."
    )

# ==================================================
# CLUSTER DISTRIBUTION
# ==================================================

st.subheader("📊 Cluster Distribution")

cluster_counts = pd.Series(
    clusters
).value_counts()

st.bar_chart(
    cluster_counts
)

# ==================================================
# CLUSTER LABELS TABLE
# ==================================================

st.subheader("📄 Cluster Labels")

df_display = df.copy()

df_display["Cluster"] = clusters

st.dataframe(
    df_display.head(50),
    use_container_width=True
)

# ==================================================
# CLUSTER SUMMARY
# ==================================================

st.subheader("📈 Cluster Summary")

summary_df = pd.DataFrame(
    cluster_counts
)

summary_df.columns = ["Customers"]

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==================================================
# NOISE ANALYSIS
# ==================================================

st.subheader("🚨 Noise Analysis")

if noise_points > 0:

    st.error(
        f"DBSCAN detected {noise_points} outlier customers."
    )

else:

    st.success(
        "No outliers detected for current settings."
    )