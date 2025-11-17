import numpy as np


class KMeans:
    def __init__(self, n_clusters,
                 max_iter):
        """Init the center of clusters.

        Parameters
        ----------
        n_clusters : number of clusters
        max_iter : number of interations
        

        Attribute
        -------
        cluster_centers_ : cluster centers
        max_iter: number of interations
        n_clusters: number of clusters
        """
        #store n_clusters + max_iter like a normal person would
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None

    

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Init the center of clusters.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        

        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers 
        """
        #basically just picking random points as the first centers
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centers = X[idx]

        return centers
        

    def fit(self, X: np.ndarray):
        """Fit with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        

        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers 
        """
        #initialize centers; use the helper from above
        self.cluster_centers_ = self._init_centers(X)
        for _ in range(self.max_iter):
            #assignment step: find nearest center for each point
            #doing euclidean distance squared bc cheaper
            distances = np.linalg.norm(
                X[:, np.newaxis] - self.cluster_centers_[np.newaxis, :],
                axis=2
            )
            labels = np.argmin(distances, axis=1)
            #recompute centers as mean of assigned points
            new_centers = []
            for k in range(self.n_clusters):
                pts = X[labels == k]
                if len(pts) == 0:
                    #if a center gets no points (sad), just keep it where it was
                    new_centers.append(self.cluster_centers_[k])
                else:
                    new_centers.append(pts.mean(axis=0))
            new_centers = np.array(new_centers)

            #stop if centers don't move anymore (basically converged)
            if np.allclose(self.cluster_centers_, new_centers):
                break

            self.cluster_centers_ = new_centers
        return self.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        

        Returns
        -------
        Labels : np.ndarray
                 Predicted labels
        """
        #compute nearest center for each new point
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_[np.newaxis, :],
            axis=2
        )
        labels = np.argmin(distances, axis=1)

        return labels
        
        

    


