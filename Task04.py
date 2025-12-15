import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Rule Extraction Playground")

st.title("Task 04: Extracting Rules from ML Models")
st.markdown("""
**The Workflow:** 
1. **Teacher:** A complex Random Forest trains on the noisy data (The Black Box).
2. **Student:** A simple Decision Tree trains to mimic the *Teacher's predictions* (The Surrogate).
3. **Extraction:** We read the IF-THEN rules directly from the Student.
""")

# --- Sidebar Controls ---
st.sidebar.header("Dataset Generation")
dataset_type = st.sidebar.selectbox("Shape", ["Moons", "Circles", "Linear"])
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2)
n_samples = st.sidebar.slider("Number of Points", 100, 1000, 500)

st.sidebar.header("Surrogate Complexity")
tree_depth = st.sidebar.slider("Max Tree Depth (Rule Complexity)", 1, 5, 3)

# --- 1. Data Generation ---
X, y = None, None
if dataset_type == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
elif dataset_type == "Circles":
    X, y = make_circles(n_samples=n_samples, noise=noise_level, factor=0.5, random_state=42)
else:
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)

# --- 2. The Black Box Model (Complex) ---
# We use a Random Forest which is hard to interpret directly
black_box = RandomForestClassifier(n_estimators=50, random_state=42)
black_box.fit(X, y)

# Get predictions from the black box (Not the ground truth!)
# We want to explain the MODEL, not the data.
y_pred_blackbox = black_box.predict(X)

# --- 3. The Surrogate Model (Simple) ---
# Train a decision tree on the Black Box's outputs
surrogate = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
surrogate.fit(X, y_pred_blackbox)

# Calculate Fidelity (How well the surrogate mimics the black box)
fidelity = surrogate.score(X, y_pred_blackbox) * 100

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Decision Boundary & Data")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display the Surrogate's boundary
    DecisionBoundaryDisplay.from_estimator(
        surrogate, X, alpha=0.3, cmap=plt.cm.RdBu, ax=ax, response_method="predict"
    )
    
    # Scatter plot of the data points colored by Black Box prediction
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred_blackbox, cmap=plt.cm.RdBu, edgecolors="k", s=25)
    ax.set_title(f"Surrogate Model Boundary (Fidelity: {fidelity:.1f}%)")
    st.pyplot(fig)

with col2:
    st.subheader("Extracted Rules")
    st.caption("These rules approximate the complex model's behavior.")
    
    # Extract text rules
    feature_names = ["X1 (x-axis)", "X2 (y-axis)"]
    rules = export_text(surrogate, feature_names=feature_names)
    
    # Basic parsing to make it look nicer
    st.code(rules, language="text")
    
    st.info(f"""
    **Interpretation:**
    If a new data point falls into these regions, the Decision Tree (Surrogate) 
    predicts the same class as the Random Forest (Black Box) {fidelity:.1f}% of the time.
    """)