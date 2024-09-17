# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

st.title("European Soccer Team Analysis Dashboard")

# Step 1: Upload CSV File
st.subheader("Upload the Player Attributes CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    team_attributes_df = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfully!")
    st.write(team_attributes_df.head())  #show the first few rows of the data

        # Define the features
    features = ['buildUpPlaySpeed', 'chanceCreationPassing',
                'chanceCreationCrossing', 'chanceCreationShooting',
                'defencePressure', 'defenceAggression', 'defenceTeamWidth']

    # Define X (features) and y (target)
    X_team = team_attributes_df[features]
    y = team_attributes_df['year']

    # --- Step 2: Feature Selection using RFE ---
    st.subheader("Feature Selection using RFE")
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=5)
    X_team_selected = rfe.fit_transform(X_team, y)

    # Display selected features
    selected_features = [features[i] for i in range(len(features)) if rfe.support_[i]]
    st.write("Selected Features:", selected_features)

    # Preprocess the data by scaling it
    scaler = StandardScaler()
    X_team_scaled = scaler.fit_transform(X_team_selected)

    """### **Hyperparameter Tuning**"""

    # Define a function to perform hyperparameter tuning
    def tune_hyperparameters(X, n_components, title):
        param_dist = {
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'max_iter': [100, 200, 300]
        }
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        random_search = RandomizedSearchCV(gmm, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42)
        random_search.fit(X)
        best_gmm = random_search.best_estimator_
        st.write(f"Best Parameters for {title} Clustering:", random_search.best_params_)
        return best_gmm

    n_components_team = 3
    st.write("Tuning Hyperparameters for Team Strategy Clustering")
    best_gmm_team = tune_hyperparameters(X_team_scaled, n_components_team, "Team Strategy")

    """### **PCA Analysis**"""
    def plot_pca_variance(X, title):
        pca = PCA()
        pca.fit(X)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
        k = np.argmax(cumulative_variance_ratio > 95) + 1
        fig, ax = plt.subplots()
        ax.plot(cumulative_variance_ratio)
        ax.axvline(x=k, color="k", linestyle="--", label=f'95% Explained Variance')
        ax.axhline(y=95, color="r", linestyle="--", label='95% Threshold')
        ax.set_title(f'Cumulative Explained Variance for {title}')
        ax.set_ylabel('Cumulative Explained Variance (%)')
        ax.set_xlabel('Principal Components')
        ax.legend()
        st.pyplot(fig)

    st.subheader("PCA Analysis")
    plot_pca_variance(X_team_scaled, "Team Strategy")

    """### **Clustering and Evaluation Metrics**"""
    def perform_em_clustering(X, gmm, title):
        labels = gmm.predict(X)
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        aic = gmm.aic(X)
        bic = gmm.bic(X)
        st.write(f"{title} Clustering - Number of Clusters: {gmm.n_components}")
        st.write(f"Silhouette Score: {silhouette}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        st.write(f"AIC: {aic}, BIC: {bic}")
        return labels

    st.subheader("Team Strategy Clustering and Metrics")
    team_labels = perform_em_clustering(X_team_scaled, best_gmm_team, "Team Strategy")

    """### **Visualization**"""
    def visualize_clusters_2d(X, labels, title):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7)
        ax.set_title(f"{title} Clustering Visualization (PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        fig.colorbar(scatter)
        st.pyplot(fig)

    st.subheader("Team Strategy Clustering Visualization")
    visualize_clusters_2d(X_team_scaled, team_labels, "Team Strategy")
