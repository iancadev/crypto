from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from ....clustering import data
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def cluster(dfs, n=100, n_clusters=3, delta=pd.Timedelta('00:30:00')):
    # Create an array to store index, start, and end row indices for each group
    groups = []
    for i, df in enumerate(dfs):
        start_id = 0
        window_length = 0
        for j in range(1, len(df)):
            if data.row_delta(df.iloc[j], df.iloc[j-1]) > delta:
                window_length = 0
                start_id = j
                continue
            window_length += 1
            if window_length >= n:
                groups.append((i, start_id, j))
                window_length = 0
                start_id = j

    opens = []
    for group in groups:
        opens.append(dfs[group[0]].iloc[group[1]:group[2]]['Open'])
    
    # Calculate pairwise Pearson sample correlations for opens
    correlation_matrix = np.corrcoef([open.values for open in opens])
    
    # Perform k-means clustering on the correlation matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Adjust n_clusters as needed
    
    # Fit the k-means model on the correlation matrix
    kmeans.fit(correlation_matrix)
    
    # Assign labels to each group
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Label rows in dfs with k-means labels and distances from centroids
    for idx, group in enumerate(groups):
        cluster_label = labels[idx]
        distances = np.linalg.norm(correlation_matrix[idx] - centroids[cluster_label])
        dfs[group[0]].loc[group[1]:group[2], 'Cluster_Label'] = cluster_label
        dfs[group[0]].loc[group[1]:group[2], 'Distance_From_Centroid'] = distances

    # Calculate clustering scores
    scores = {
        "silhouette_score": silhouette_score(correlation_matrix, labels, metric='euclidean'),
        "calinski_harabasz_score": calinski_harabasz_score(correlation_matrix, labels),
        "davies_bouldin_score": davies_bouldin_score(correlation_matrix, labels)
    }

    return scores
