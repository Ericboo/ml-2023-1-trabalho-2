import numpy as np

def dropLabel(dataset):
    new_dataset = np.delete(dataset, 1, axis=1)
    new_dataset = np.array(new_dataset).astype(int)
    return new_dataset

def subdivide(dataset, divisao):
    if divisao == 'treinamento':
        treinamento_positivos = dataset[:100]
        treinamento_negativos = dataset[600:1001]        
        return np.concatenate((treinamento_positivos, treinamento_negativos), axis=0)
    elif divisao == 'teste':
        teste_negativos = dataset[1001:2000]
        teste_positivos = dataset[101:500]
        teste = np.concatenate((teste_positivos, teste_negativos), axis=0)
        return teste
    elif divisao == 'label treinamento':
        label = []
        treinamento = subdivide(dataset, 'treinamento')
        for i in range(len(treinamento)):
           label.append(treinamento[i][1])
        return label
    elif divisao == 'label teste':
        label = []
        teste = subdivide(dataset, 'teste')
        for i in range(len(teste)):
           label.append(teste[i][1])
        return label

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, n_features, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        feature_value = x[node.feature]
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            thresholds = np.unique(vals)
            for threshold in thresholds:
                gain = self._information_gain(np.array(y), vals, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_feat, threshold):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_feat, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])       
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X, threshold):
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y.astype(int)).argmax()
    
def entropy(y):
    hist = np.bincount(np.array(y).astype(int))
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def execute(dataset):
    X = subdivide(dataset, 'treinamento')
    y = subdivide(dataset, 'label treinamento')
    X = dropLabel(X)
    tree = DecisionTree(max_depth=1)
    tree.fit(X, np.array(y))
    X = subdivide(dataset, 'teste')
    y = subdivide(dataset, 'label teste')
    X = dropLabel(X)
    preds = tree.predict(X)
    #corretas = np.sum(np.array(preds).astype(int) == np.array(y).astype(int))
    #print("De 1398 pacientes, a árvore de decisão preveu corretamente:", corretas)
    return (y, preds)