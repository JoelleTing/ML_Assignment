# importing the necessary libraries
# data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.title("European Soccer Team Analysis Dashboard")

# Step 1: Upload CSV File
st.subheader("Upload the Player Attributes CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    team_attributes_df = pd.read_csv(uploaded_file)

"""## **Expectation–Maximization Algorithm (EM)**"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sqlite3
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from google.colab import drive
drive.mount('/content/drive/')

team_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/European Soccer/team_att_merged.csv')

# Define the features
features = ['buildUpPlaySpeed', 'chanceCreationPassing',
            'chanceCreationCrossing', 'chanceCreationShooting',
            'defencePressure', 'defenceAggression', 'defenceTeamWidth']

# Define X (features) and y (target)
X_team = team_data[features]
y = team_data['year']

# --- Step 2: Feature Selection using RFE ---
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_team_selected = rfe.fit_transform(X_team, y)

# Print selected features and ranking
selected_features = [features[i] for i in range(len(features)) if rfe.support_[i]]
print("Selected Features:", selected_features)

"""### **Scaling**"""

# Preprocess the data by scaling it
scaler = StandardScaler()
X_team_scaled = scaler.fit_transform(X_team_selected)      # Team strategy data

"""### **Hyperparameter Tuning**"""

# Define a function to perform hyperparameter tuning with a fixed number of components
def tune_hyperparameters(X, n_components, title):
    param_dist = {
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'max_iter': [100, 200, 300]
    }

    # Initialize GaussianMixture with fixed n_components
    gmm = GaussianMixture(n_components=n_components, random_state=42)

    # Perform random search for hyperparameter tuning
    random_search = RandomizedSearchCV(gmm, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42)
    random_search.fit(X)

    best_gmm = random_search.best_estimator_
    print(f"Best Parameters for {title} Clustering:", random_search.best_params_)
    return best_gmm

# Perform hyperparameter tuning with n_components fixed at 3
n_components_team = 3

print("Tuning Hyperparameters for Team Strategy Clustering")
best_gmm_team = tune_hyperparameters(X_team_scaled, n_components_team, "Team Strategy")

"""### **Principal Component Analysis (PCA)**"""

def plot_pca_variance(X, title):
    # Initialize PCA
    pca = PCA()
    pca.fit(X)

    # Calculate cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"Cumulative Explained Variance Ratio for {title}:")
    print(cumulative_variance_ratio)

    # Print number of components explaining 95% variance
    k = np.argmax(cumulative_variance_ratio > 95) + 1  # +1 to get the actual number of components
    print(f"Number of components explaining 95% variance for {title}: {k}")

    # Plot cumulative explained variance
    plt.figure(figsize=[10,5])
    plt.title(f'Cumulative Explained Variance for {title}')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.xlabel('Principal Components')
    plt.axvline(x=k, color="k", linestyle="--", label=f'95% Explained Variance')
    plt.axhline(y=95, color="r", linestyle="--", label='95% Threshold')
    plt.plot(cumulative_variance_ratio, label='Cumulative Explained Variance')
    plt.legend()
    plt.show()

# Plot for Team Strategy data
plot_pca_variance(X_team_scaled, "Team Strategy")

"""### **Clustering & Evaluation Metrics**"""

# Define a function to perform EM clustering and evaluate metrics
def perform_em_clustering(X, gmm, title):
    labels = gmm.predict(X)

    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    aic = gmm.aic(X)
    bic = gmm.bic(X)

    print(f"{title} Clustering - Number of Clusters: {gmm.n_components}")
    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print("-" * 50)

    return labels

# Perform EM clustering and evaluate for Team Strategy
print("Evaluating Team Strategy Clustering")
team_labels = perform_em_clustering(X_team_scaled, best_gmm_team, "Team Strategy")

"""### **Validation**"""

# Define a function for cross-validation of EM clustering with additional metrics
def cross_validate_em(X, n_components, covariance_type, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]

        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        gmm.fit(X_train)

        labels_test = gmm.predict(X_test)

        silhouette = silhouette_score(X_test, labels_test)
        davies_bouldin = davies_bouldin_score(X_test, labels_test)
        calinski_harabasz = calinski_harabasz_score(X_test, labels_test)

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski_harabasz)

    avg_silhouette = np.mean(silhouette_scores)
    avg_davies_bouldin = np.mean(davies_bouldin_scores)
    avg_calinski_harabasz = np.mean(calinski_harabasz_scores)

    print(f"Cross-Validated Silhouette Score: {avg_silhouette}")
    print(f"Cross-Validated Davies-Bouldin Index: {avg_davies_bouldin}")
    print(f"Cross-Validated Calinski-Harabasz Index: {avg_calinski_harabasz}")

    return avg_silhouette, avg_davies_bouldin, avg_calinski_harabasz

# Validate Team Strategy Clustering
print("\nValidating Team Strategy Clustering:")
cross_validate_em(X_team_scaled, n_components_team, best_gmm_team.covariance_type)

"""### **Visualization**"""

def visualize_clusters_2d(X, labels, title):
    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a scatter plot of the clusters
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.7)

    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} Clustering Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Visualize Team Strategy Clustering
visualize_clusters_2d(X_team_scaled, team_labels, "Team Strategy")

def plot_pairwise_clusters(X, labels, title):
    # Convert labels to a DataFrame for easier plotting
    df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    df['Cluster'] = labels

    # Plot pairwise features with hue as the cluster labels
    sns.pairplot(df, hue='Cluster', palette='viridis', markers='o', diag_kind='kde')
    plt.suptitle(f"{title} Pairwise Feature Distribution", y=1.02)
    plt.show()

# Plot Pairwise Features for Team Strategy Clustering
plot_pairwise_clusters(X_team_scaled, team_labels, "Team Strategy")

def plot_cluster_centroids(gmm, X, title):
    # Get means of Gaussian components
    means = gmm.means_

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    means_pca = pca.transform(means)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=team_labels, palette='viridis', marker='o', s=50, alpha=0.7)
    plt.scatter(means_pca[:, 0], means_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(f"{title} Clustering with Centroids (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

# Plot Cluster Centroids for Team Strategy Clustering
plot_cluster_centroids(best_gmm_team, X_team_scaled, "Team Strategy")

from sklearn.manifold import TSNE

def visualize_clusters_tsne(X, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', marker='o', s=50, alpha=0.7)
    plt.title(f"{title} Clustering Visualization (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title='Cluster')
    plt.show()

# Visualize with t-SNE
visualize_clusters_tsne(X_team_scaled, team_labels, "Team Strategy")
