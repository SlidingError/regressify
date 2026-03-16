import numpy as np
class Node():
    def __init__(self, feature=None, thresh=None, left=None, right=None, value=None):
        self.feature = feature
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value

class RegressionTree():
    def __init__(self, max_depth=3, min_samples=3):
        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def fit(self, X, y):
        dataset = np.column_stack((X, y))
        self.root = self._build_tree(dataset)

    def _build_tree(self, dataset, depth=0):
        X = dataset[:,:-1]
        y = dataset[:, -1]

        n_samples, n_features = X.shape

        # Stop splitting when one of the stopping conditions is met
        # - too few samples, reached max depth, or all targets are identical
        if n_samples < self.min_samples or depth >= self.max_depth or len(np.unique(y)) == 1:
            return Node(value=self._evaluate(y))
        
        best_feature, best_thresh, best_error = None, None, None
        best_left, best_right = None, None

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            for thresh in values:
                left = dataset[X[:, feature] <= thresh]
                right = dataset[X[:, feature] > thresh]

                if len(left) == 0 or len(right) == 0:
                    continue

                error = self._error(left[:,-1], right[:, -1])
                if best_error is None or error < best_error:
                    best_feature = feature
                    best_thresh = thresh
                    best_left = left
                    best_right = right
                    best_error = error
        
        if best_error is None:
            return Node(value=self._evaluate(y))
        
        left_node = self._build_tree(best_left, depth+1)
        right_node = self._build_tree(best_right, depth+1)

        return Node(best_feature, best_thresh, left_node, right_node)
    
    def _evaluate(self, y):
        return np.mean(y)
    
    def _error(self, left, right):
        return np.sum((left - np.mean(left)) ** 2) + np.sum((right - np.mean(right)) ** 2)
    
    def predict(self, X):
        """Predict target values for the given input samples.

        X can be a 2D array-like (n_samples, n_features) or a 1D array-like for a single sample.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._predict(X)

    def _predict(self, X):
        return np.array([self._predict_by_row(row, self.root) for row in X])

    def _predict_by_row(self, x, node: Node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.thresh:
            return self._predict_by_row(x, node.left)
        return self._predict_by_row(x, node.right)
