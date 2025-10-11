
import numpy as np
import pandas as pd


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None


class MyTreeReg:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leaves: int = 20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaves = max_leaves
        self.leafs_cnt = None
        self.bins = bins
        self.tree = None
        self.leafs_cnt = 1
        self.feature_splits = {}
        self.fi = {}
        self.N = None
        self.sum_tree_values = 0

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series):
        col_name = None
        split_value = None
        best_mse = float('-inf')

        mse_p = self._mse(y)
        for feature in range(X.shape[1]):
            if self.bins is not None:
                thresholds = self.feature_splits[feature]
                if len(thresholds) == 0:
                    continue
            else:
                thresholds = np.unique(sorted(X.iloc[:, feature]))
                thresholds = [(thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds) - 1)]

            n = len(y)
            for thresh in thresholds:
                left = X.iloc[:, feature] <= thresh
                right = X.iloc[:, feature] > thresh

                if sum(left) == 0 or sum(right) == 0:
                    continue

                mse_left = self._mse(y[left])
                mse_right = self._mse(y[right])

                n_left = len(y[left])
                n_right = len(y[right])

                mse_t = mse_p - (n_left / n * mse_left + n_right / n * mse_right)
                if mse_t > best_mse:
                    col_name = X.columns[feature]
                    split_value = thresh
                    best_mse = mse_t
        return col_name, split_value, best_mse

    def _calculate_feature_splits(self, X: pd.DataFrame):
        for feature in range(X.shape[1]):
            unique_values = np.unique(X.iloc[:, feature])
            if len(unique_values) <= self.bins - 1:
                self.feature_splits[feature] = unique_values
            else:
                _, bin_edges = np.histogram(X.iloc[:, feature], bins=self.bins)
                self.feature_splits[feature] = bin_edges[1:-1]

    def fit(self, X: pd.DataFrame, y: pd.Series, N=None):
        self.tree = None
        if self.N is None:
            self.N = X.shape[0]
        else:
            self.N = N
        self.fi = {col: 0 for col in X.columns}
        if self.bins is not None:
            self._calculate_feature_splits(X)

        self.tree = self._build_tree(self.tree, X, y)

    def _build_tree(self, root, X_node, y_node, side='root', depth=0):
        if root is None:
            root = Node()

        if len(y_node.unique()) == 1 or depth >= self.max_depth or \
                len(y_node) < self.min_samples_split or \
                (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leaves):
            root.side = side
            root.value_leaf = np.mean(y_node)
            self.sum_tree_values += root.value_leaf
            return root

        col_name, best_threshold, gain = self._get_best_split(X_node, y_node)

        if best_threshold is None:
            root.side = side
            root.value_leaf = np.mean(y_node)
            self.sum_tree_values += root.value_leaf
            return root

        root.feature = col_name
        root.value_split = best_threshold
        self.leafs_cnt += 1

        self.fi[col_name] += gain * len(y_node) / self.N

        left_indices = X_node[col_name] <= best_threshold
        right_indices = X_node[col_name] > best_threshold

        X_left, y_left = X_node.loc[left_indices], y_node[left_indices]
        X_right, y_right = X_node.loc[right_indices], y_node[right_indices]

        root.left = self._build_tree(root.left, X_left, y_left, 'left', depth + 1)
        root.right = self._build_tree(root.right, X_right, y_right, 'right', depth + 1)

        return root

    def predict(self, X: pd.DataFrame):
        logits = [self._predict_single_proba(row, self.tree) for _, row in X.iterrows()]
        return np.array(logits)

    def _predict_single_proba(self, x, tree):
        if tree.value_leaf is not None:
            return tree.value_leaf
        if x[tree.feature] <= tree.value_split:
            return self._predict_single_proba(x, tree.left)
        else:
            return self._predict_single_proba(x, tree.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{'1.' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{'1.' * depth}{node.side} = {node.value_leaf}")
