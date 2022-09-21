import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node), -1 means the node is not a leaf node
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample till reaching a leaf node
            root = self.root
            while(root.right_child != None and root.left_child != None):
                if(test_x[i][root.feature] == 1):
                    root = root.right_child
                elif(test_x[i][root.feature] == 0):
                    root = root.left_child
            val, count = np.unique(root.label, return_counts = True)
            prediction[i] = val[np.argmax(count)]

        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node based on minimum node entropy (if yes, find the corresponding class label with majority voting and exit the current recursion)
        if(self.min_entropy > node_entropy):
            val, count = np.unique(label, return_counts = True)
            cur_node.label = val[np.argmax(count)]
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        cur_node.right_child = self.generate_tree(data[data[:, selected_feature] == 1], label[data[:, selected_feature] == 1])
        cur_node.left_child = self.generate_tree(data[data[:, selected_feature] == 0], label[data[:, selected_feature] == 0])

        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        entropy = np.zeros(len(data[0]))
        for i in range(len(data[0])):

            # compute the entropy of splitting based on the selected features
            entropy[i] = self.compute_split_entropy(label[data[:,i] == 0], label[data[:,i] == 1])

            # select the feature with minimum entropy
            
        return np.argmin(entropy)

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y are labels for the two branches
        split_entropy = -1 # placeholder
        split_entropy = (len(left_y) / (len(left_y) + len(right_y))) * self.compute_node_entropy(left_y) + (len(right_y) / (len(left_y) + len(right_y))) * self.compute_node_entropy(right_y)
        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = -1 # placeholder
        node_entropy += 1
        for i in np.unique(label):
            p = (label == i).sum() / len(label)
            node_entropy -= p * np.log2(p + 1e-15)
        return node_entropy
