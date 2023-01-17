import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances


class KMeans:
    def __init__(self, k: int = 5, threshold: float = 0.01):
        self.k = k
        self.rss_threshold = threshold
        self.clusters = []
        self.assignments = []
        self.history = [float('inf')]

    def _assign_clusters(self, X, clusters):


    def fit(self, X: np.ndarray):
        """
        Find the best k centroids with a minimum RSS possible
        Steps:
            - initialize clusters and assignment
            - repeat until the stop condition met:
                - assign points to the clusters
                - compute RSS
                - check stop condition
                - compute new clusters
        :param X: training data
        :return: self object: computed estimator
        """

        # Initialize clusters
        n_samples, n_features = X.shape[0], X.shape[1]
        random_sample_indices = random.sample(list(range(n_samples)), self.k)
        self.clusters = X[random_sample_indices]

        # Run the algorithm until the stop condition met
        current_rss = self.history[0] - 1 - self.rss_threshold
        while self.history[-1] - current_rss > self.rss_threshold:



