import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .prediction-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 30px; border-radius: 15px; color: white;
            text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .cluster-info {
            background-color: rgba(255,255,255,0.95);
            padding: 20px; border-radius: 10px;
            margin: 10px 0; border-left: 5px solid #11998e;
        }
        .algo-badge {
            background: #11998e; color: white; padding: 5px 15px;
            border-radius: 20px; font-size: 13px; font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

CLUSTER_INFO = {
    0: {"name": "High Value Customers",   "color": "#FF6B6B",
        "characteristics": ["Age: Young (25-40)", "Income: High (40-80k)", "Spending: High (70-100)"],
        "description": "Young customers with high spending score and good income"},
    1: {"name": "Potential Target",        "color": "#4ECDC4",
        "characteristics": ["Age: Middle-aged (35-50)", "Income: Moderate (30-70k)", "Spending: Moderate-High"],
        "description": "Middle-aged customers with moderate to high spending"},
    2: {"name": "Average Customers",       "color": "#45B7D1",
        "characteristics": ["Age: Young (20-50)", "Income: Low (20-50k)", "Spending: Low-Moderate"],
        "description": "Young customers with low to moderate spending"},
    3: {"name": "Loyal Customers",         "color": "#FFA07A",
        "characteristics": ["Age: Older (40-70)", "Income: Variable", "Spending: Variable"],
        "description": "Older customers with variable spending patterns"},
    4: {"name": "Budget Conscious",        "color": "#98D8C8",
        "characteristics": ["Age: Varied", "Income: High (50-150k)", "Spending: Low (10-50)"],
        "description": "Customers with high income but low spending"},
}

RECOMMENDATIONS = {
    0: "🎯 Premium targeting strategy — Focus on retention and upselling premium products",
    1: "📈 Growth opportunity — Target with seasonal promotions and loyalty programs",
    2: "🎁 Budget offerings — Create value packs and discounts to increase engagement",
    3: "🤝 Relationship building — Personalized communication and special offers",
    4: "💎 Exclusive products — Premium/investment products despite lower spending",
}

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

@st.cache_resource
def train_hierarchical(n_clusters, linkage_method):
    df = load_data()
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(scaled)
    return model, scaler, labels, scaled

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<span class='algo-badge'>Hierarchical Clustering</span>", unsafe_allow_html=True)
st.title("🌿 Mall Customer Segmentation — Hierarchical Clustering")
st.markdown("Build a **tree of clusters (dendrogram)** using bottom-up agglomerative approach.")
st.markdown("---")

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Hierarchical Settings")
    n_clusters      = st.slider("Number of Clusters", 2, 10, 5)
    linkage_method  = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    st.markdown("---")
    st.markdown("**How Hierarchical works:**")
    st.markdown("""
- Start: each point is its own cluster  
- Merge the two closest clusters  
- Repeat until K clusters remain  
- Cut dendrogram at desired level  
    """)

model, scaler, labels, scaled = train_hierarchical(n_clusters, linkage_method)

# ── Input + Stats ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.markdown("### 📊 Customer Information")
    age            = st.slider("👤 Age",                     int(df['Age'].min()),               int(df['Age'].max()),               30)
    annual_income  = st.slider("💰 Annual Income (k$)",     int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 50)
    spending_score = st.slider("🎯 Spending Score (1-100)",  1, 100, 50)
    ca, cb, cc = st.columns(3)
    ca.metric("Age",      f"{age} yrs")
    cb.metric("Income",   f"${annual_income}k")
    cc.metric("Spending", f"{spending_score}/100")

with col2:
    st.markdown("### 📈 Dataset Statistics")
    cx, cy, cz = st.columns(3)
    cx.metric("Total Customers", len(df))
    cy.metric("Avg Age",    f"{df['Age'].mean():.1f} yrs")
    cz.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    st.markdown("---")
    cp, cq, cr = st.columns(3)
    cp.metric("Min Income",   f"${df['Annual Income (k$)'].min():.0f}k")
    cq.metric("Max Income",   f"${df['Annual Income (k$)'].max():.0f}k")
    cr.metric("Avg Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🚀 Predict Cluster", use_container_width=True):
    inp        = np.array([[age, annual_income, spending_score]])
    scaled_inp = scaler.transform(inp)

    # Find nearest point in scaled space
    dists   = np.linalg.norm(scaled - scaled_inp, axis=1)
    nearest = int(np.argmin(dists))
    cluster = int(labels[nearest])

    safe_cluster = cluster if cluster in CLUSTER_INFO else 0
    details = CLUSTER_INFO[safe_cluster]

    st.markdown(f"""
        <div class="prediction-box">
            <h2 style='margin:0;font-size:26px;'>Hierarchical Cluster Prediction</h2>
            <h1 style='margin:10px 0;font-size:52px;'>Cluster {cluster}</h1>
            <h3 style='font-size:22px;'>{details['name']}</h3>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    ci1, ci2 = st.columns(2, gap="large")
    with ci1:
        st.markdown("### 📋 Cluster Description")
        st.markdown(f"<div class='cluster-info'><p><strong>{details['description']}</strong></p></div>", unsafe_allow_html=True)
        st.markdown("### 🎯 Key Characteristics")
        for c in details['characteristics']:
            st.markdown(f"• {c}")
    with ci2:
        st.markdown("### 💡 Cluster Insights")
        mask = labels == cluster
        sub  = df[mask]
        st.markdown(f"""
            <div class='cluster-info'>
                <p><strong>Size:</strong> {mask.sum()} customers ({mask.sum()/len(df)*100:.1f}%)</p>
                <p><strong>Avg Age:</strong> {sub['Age'].mean():.1f} yrs</p>
                <p><strong>Avg Income:</strong> ${sub['Annual Income (k$)'].mean():.1f}k</p>
                <p><strong>Avg Spending:</strong> {sub['Spending Score (1-100)'].mean():.1f}/100</p>
            </div>""", unsafe_allow_html=True)

    # ── Visualisations ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Visualizations")
    vc1, vc2 = st.columns(2, gap="large")

    with vc1:
        fig = px.scatter_3d(
            x=df['Age'], y=df['Annual Income (k$)'], z=df['Spending Score (1-100)'],
            color=labels.astype(str),
            labels={'x': 'Age', 'y': 'Annual Income (k$)', 'z': 'Spending Score'},
            title='3D Cluster Distribution (Hierarchical)',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.add_scatter3d(
            x=[age], y=[annual_income], z=[spending_score],
            mode='markers', marker=dict(size=14, color='red', symbol='diamond'),
            name='Your Input'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with vc2:
        counts = pd.Series(labels).value_counts().sort_index()
        cluster_names = [CLUSTER_INFO.get(i, {}).get('name', f'Cluster {i}') for i in counts.index]
        colors_list   = [CLUSTER_INFO.get(i, {}).get('color', '#888888') for i in counts.index]
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"Cluster {i}: {n}" for i, n in zip(counts.index, cluster_names)],
            values=counts.values, marker=dict(colors=colors_list), textposition='inside'
        )])
        fig_pie.update_layout(title='Customer Distribution by Cluster', height=500)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Dendrogram
    st.markdown("### 🌳 Dendrogram (sample of 50 points)")
    sample_idx = np.random.choice(len(scaled), min(50, len(scaled)), replace=False)
    Z = linkage(scaled[sample_idx], method=linkage_method)
    fig_d, ax = plt.subplots(figsize=(14, 5))
    ax.set_facecolor('#f8f9fa')
    fig_d.patch.set_facecolor('#f8f9fa')
    dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]),
               leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)', fontsize=13)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close()

    st.info(f"**Recommendation:** {RECOMMENDATIONS.get(safe_cluster, '—')}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>🌿 Hierarchical Clustering | Mall Customer Segmentation</div>", unsafe_allow_html=True)