# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

st.title("European Soccer Team Analysis Dashboard")

# Step 1: Upload CSV File
st.subheader("Upload the Player Attributes CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    team_attributes_df = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfully!")
    st.write(team_attributes_df.head())  # Show the first few rows of the data

     # Let the user choose the features they want to include in the analysis
    available_features = ['buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing', 'chanceCreationCrossing', 
                          'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth']

    selected_features = st.multiselect("Select the features you want to include:", available_features, default=available_features)

    if selected_features:
        # Define X (features) and y (target)
        X_team = team_attributes_df[selected_features]
        y = team_attributes_df['year']

        # --- Step 2: Feature Selection using RFE ---
        st.subheader("Feature Selection using RFE")
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=min(5, len(selected_features)))  # Adjust the number of selected features
        X_team_selected = rfe.fit_transform(X_team, y)

        # Display selected features
        selected_rfe_features = [selected_features[i] for i in range(len(selected_features)) if rfe.support_[i]]
        st.write("Selected Features:", selected_rfe_features)

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

        # Step 3: Let user define number of components (with default value)
        st.subheader("Define Number of Components for Clustering")
        n_components_team = st.number_input("Select number of components for Team Strategy Clustering:", 
                                            min_value=2, max_value=10, value=3, step=1)
        
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

         # Visualization - Pairwise Clusters using Matplotlib
        def plot_pairwise_clusters(X, labels, title):
            df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
            df['Cluster'] = labels

            # Plot pairwise scatter plots using Matplotlib
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            axes = axes.flatten()
            for i in range(len(df.columns) - 1):
                for j in range(i + 1, len(df.columns) - 1):
                    axes[i].scatter(df.iloc[:, i], df.iloc[:, j], c=labels, cmap='viridis', marker='o', alpha=0.7)
                    axes[i].set_xlabel(f"Feature {i+1}")
                    axes[i].set_ylabel(f"Feature {j+1}")
                    axes[i].set_title(f"Feature {i+1} vs Feature {j+1}")
            fig.suptitle(f"{title} Pairwise Feature Distribution", y=1.02)
            st.pyplot(fig)

        st.subheader("Team Strategy Pairwise Feature Distribution")
        plot_pairwise_clusters(X_team_scaled, team_labels, "Team Strategy")

         # Visualization - Cluster Centroids using Matplotlib
        def plot_cluster_centroids(gmm, X, labels, title):
            means = gmm.means_

            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            means_pca = pca.transform(means)

            plt.figure(figsize=(10, 7))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7)
            plt.scatter(means_pca[:, 0], means_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
            plt.title(f"{title} Clustering with Centroids (PCA)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            st.pyplot(plt)

        st.subheader("Team Strategy Clustering with Centroids")
        plot_cluster_centroids(best_gmm_team, X_team_scaled, team_labels, "Team Strategy")

        # Visualization - t-SNE using Matplotlib
        def visualize_clusters_tsne(X, labels, title):
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(10, 7))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7)
            plt.title(f"{title} Clustering Visualization (t-SNE)")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.colorbar(label='Cluster')
            st.pyplot(plt)

        st.subheader("Team Strategy Clustering Visualization (t-SNE)")
        visualize_clusters_tsne(X_team_scaled, team_labels, "Team Strategy")
